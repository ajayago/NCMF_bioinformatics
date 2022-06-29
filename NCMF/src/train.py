import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch import autograd
from scipy import sparse
from collections import defaultdict
from src.models import *
from src.utils import *
from src.loss import *
from src.data_utils import *
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def train_and_validate(net, trainloaders, validloaders, embloaders, norm_params, hyperparams, device, writer, matrix_types):
    """Trains and validates the given network."""
    optimizer = optim.AdamW(net.parameters(
    ), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'], eps=1e-10, amsgrad=False)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, hyperparams['num_epochs'], eta_min=0, last_epoch=-1, verbose=False)
    T_0 = hyperparams['num_epochs'] // hyperparams['num_cycles']
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=1)
    train_loss_func = [make_train_loss_function_ZINB(),make_train_loss_function_ZINORM()]
    valid_loss_func = [make_valid_loss_function_ZINB(),make_valid_loss_function_ZINORM()]
    scaler = amp.GradScaler()

    train_losses, valid_losses = [], []
    rmse_epochs = {}
    for epoch in range(hyperparams['num_epochs']):
        net.train()
        beta = anneal_beta(
            hyperparams['num_cycles'], 
            hyperparams['proportion'], 
            hyperparams['num_epochs'], 
            epoch, 
            hyperparams['anneal']
        )
        train_loss, train_rmse_loss = train_step(
            net,
            optimizer,
            scheduler,
            scaler,
            train_loss_func,
            trainloaders,
            norm_params,
            hyperparams,
            beta,
            device,
            matrix_types
        )
        train_losses.append(train_loss)

        net.eval()
        with torch.no_grad():
            entity_embedding = retrieve_embedding(
                net, 
                embloaders, 
                norm_params, 
                device
            )
            valid_loss, valid_rmse_loss = valid_step(
                net,
                valid_loss_func,
                validloaders,
                entity_embedding,
                norm_params,
                hyperparams,
                device,
                matrix_types
            )
            valid_losses.append(valid_loss)

        if torch.isnan(torch.tensor(train_loss)):
            print('NaN')
            break

        print(f'====> Epoch {epoch}: Average Train Loss: {train_loss:.7f} | Train RMSE: {train_rmse_loss:.7f} | Average Valid Loss: {valid_loss:.7f} | Valid RMSE: {valid_rmse_loss:.7f} | beta: {beta}')
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/valid', valid_loss, epoch)
        writer.add_scalar('rmse/train', train_rmse_loss, epoch)
        writer.add_scalar('rmse/valid', valid_rmse_loss, epoch)
        
#        # reconstructing matrices after every 500 epochs and getting RMSE test scores
#        if (epoch + 1) % 500 == 0:
#            XP, _, _, _, _ = reconstruct(net, trainloaders, norm_params, device)
#            XP_flattened = MinMaxScaler().fit_transform(XP["X0"]).flatten()
#            Xtest_flattened = Xtest.flatten()
#            nz_list_pred = []
#            nz_list_test = []
#            for i in idx:
#                nz_list_pred.append(XP_flattened[i])
#                nz_list_test.append(Xtest_flattened[i])
#            rmse = np.sqrt(mean_squared_error(nz_list_test,nz_list_pred))
#            print(f"RMSE at {epoch} epochs = {rmse}")
#            rmse_epochs[epoch] = rmse

        if epoch != 0 and abs(train_losses[epoch] - train_losses[epoch - 1]) < hyperparams['convergence_threshold']:
            print('Convergence')
            break

    print('Finished Training\n')
    losses = {
        'train_loss': train_losses,
        'valid_loss': valid_losses,
        'train_rmse': train_rmse_loss,
        'valid_rmse': valid_rmse_loss,
    }
#    print("RMSE per 500 epochs:")
#    print(rmse_epochs)
    return net, losses


def train_step(net, optimizer, scheduler, scaler, loss_func, trainloaders, norm_params, hyperparams, beta, device, matrix_types):
    """Trains the network on the training set for a single epoch."""
    rmse_loss_func = RMSELoss(reduction='mean')

    total_loss, total_rmse = 0., 0.
    for xid, trainloader in trainloaders.items():
        row_ids, row_loaders = trainloader[0]
        col_ids, col_loaders = trainloader[1]
        # check if input is real/binary - add condition for this. right now adapted for NUHS
        if xid in matrix_types["real"]: # ZINORM
            #print("ZINORM")
            updated_loss_func = loss_func[1]
        elif xid in matrix_types["binary"]: # ZINB
            #print("ZINB")
            updated_loss_func = loss_func[0]

        for row_idx, mat_batches in enumerate(zip(*row_loaders)):
            Xs = [mat[0].to(device) for mat in mat_batches]
            X_masks = [mat[1].to(device) for mat in mat_batches]
            X_bars = [Normalise(*norm_params)(X) for X in Xs]

            col_coord = (0, 0)
            for col_idx, trans_batches in enumerate(zip(*col_loaders)):
                optimizer.zero_grad()

                XTs = [trans[0].to(device) for trans in trans_batches]
                XT_masks = [trans[1].to(device) for trans in trans_batches]
                XT_bars = [Normalise(*norm_params)(XT) for XT in XTs]
                # first item is the matirx to reconstruct
                X_block, _ = get_block(col_coord, Xs[0], XTs[0])
                X_block_mask, col_coord = get_block(
                    col_coord, X_masks[0], XT_masks[0])

                with amp.autocast():
                    XP_block, row_entities, col_entities = net(
                        X_bars, XT_bars, row_ids, col_ids)
                    XP_block = {k: b.view(X_block.shape)
                                for k, b in XP_block.items()}

                    row_loss, col_loss, rec_loss = updated_loss_func(
                        row_entities, col_entities, XP_block,
                        Xs, XTs, X_block,
                        torch.tensor(hyperparams['lamda']),
                        X_masks, XT_masks, X_block_mask,
                        beta
                    )

                    loss = row_loss + col_loss + rec_loss

                # work around for amp + checkpointing
                scaler.scale(loss).backward()

                with torch.no_grad():
                    rmse_loss = rmse_loss_func(
                        XP_block['M_bar'], X_block, X_block_mask)

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    net.parameters(), hyperparams['max_norm'])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                loss = row_loss.item() + col_loss.item() + rec_loss.item()
                total_loss += loss
                total_rmse += rmse_loss

                # print(f'{row_idx}/{len(trainloader[0][1][0])} | {col_idx}/{len(trainloader[1][1][0])} | {loss}')

    return total_loss, total_rmse


def valid_step(net, loss_func, dataloaders, entity_embedding, norm_params, hyperparams, device, matrix_types):
    """Validates the network with an unseen validation set for a single epoch."""
    rmse_loss_func = RMSELoss(reduction='mean')

    total_loss, total_rmse = 0., 0.
    for xid, dataloader in dataloaders.items():
        row_eid, col_eid = net.meta[xid]
        if xid in matrix_types["real"]: # ZINORM
            print("ZINORM")
            updated_loss_func = loss_func[1]
        elif xid in matrix_types["binary"]: # ZINB
            print("ZINB")
            updated_loss_func = loss_func[0]
        for i, (row, col, val) in enumerate(dataloader):
            row_emb = entity_embedding[row_eid][row].to(device)
            col_emb = entity_embedding[col_eid][col].to(device)
            X_block = val.view((-1, 1)).to(device)
            with amp.autocast():
                XP_block = net.rec_forward(xid, row_emb, col_emb)
                zinb_loss = updated_loss_func(
                    XP_block['M_bar'], XP_block['Theta'], XP_block['Pi'], X_block, hyperparams['lamda'], mask=None)
                rmse_loss = rmse_loss_func(XP_block['M_bar'], X_block, None)

                total_loss += zinb_loss.item()
                total_rmse += rmse_loss.item()
    return total_loss, total_rmse


def reconstruct(net, trainloaders, norm_params, device):
    """Reconstructs all the predicted matrices and learned embeddings.
    
    XP: recontructed matrix from learned embeddings from fusion of embeddings.
    M_bar: reconstructed matrix from learned embeddings (mean) from autoencoder.
    mu: learned embeddings from the mean layer of the autoencoder

    NOTE: This is not the function used in dCMF++. This function is used to visualise
    output of matrices reconstructed in particular the EmbedMNIST dataset.
    """
    print('Reconstruct')
    net.eval()
    XP, row_M_bar, col_M_bar, row_mu, col_mu = {}, {}, {}, {}, {}
    for xid, trainloader in trainloaders.items():
        with torch.no_grad():
            XP[xid], row_M_bar[xid], col_M_bar[xid], row_mu[xid], col_mu[xid] = reconstruct_step(
                net, trainloader, norm_params, device)
    return XP, row_M_bar, col_M_bar, row_mu, col_mu


def reconstruct_step(net, dataloaders, norm_params, device):
    """"Reconstructs the predicted matrices and learned embeddings of a single matrix."""
    row_ids, row_loaders = dataloaders[0]
    col_ids, col_loaders = dataloaders[1]

    nrows = len(dataloaders[0][1][0])
    ncols = len(dataloaders[1][1][0])

    XP = []
    row_M_bars = [[] for i in range(len(row_loaders))]
    col_M_bars = [[] for i in range(len(col_loaders))]
    row_mus = [[] for i in range(len(row_loaders))]
    col_mus = [[] for i in range(len(col_loaders))]

    for row_idx, mat_batches in enumerate(zip(*row_loaders)):
        XP_row = []
        Xs = [mat[0].to(device) for mat in mat_batches]
        X_bars = [Normalise(*norm_params)(X) for X in Xs]

        col_coord = (0, 0)
        for col_idx, trans_batches in enumerate(zip(*col_loaders)):
            XTs = [trans[0].to(device) for trans in trans_batches]
            XT_bars = [Normalise(*norm_params)(XT) for XT in XTs]

            X_block, col_coord = get_block(col_coord, X_bars[0], XT_bars[0])
            with amp.autocast():
                XP_block, row_entities, col_entities = net(
                    X_bars, XT_bars, row_ids, col_ids)
            XP_block = {k: b.view(X_block.shape) for k, b in XP_block.items()}

            XP_row += [XP_block['M_bar'].detach().cpu()]
            if row_idx == nrows - 1:  # last row
                for col_entity, col_M_bar, col_mu in zip(col_entities, col_M_bars, col_mus):
                    col_M_bar += [col_entity['M_bar'].detach().cpu()]
                    col_mu += [col_entity['mu'].detach().cpu()]

            print(f'{row_idx}/{nrows} | {col_idx}/{ncols}')

        XP += [torch.cat(XP_row, 1)]
        for row_entity, row_M_bar, row_mu in zip(row_entities, row_M_bars, row_mus):
            row_M_bar += [row_entity['M_bar'].detach().cpu()]
            row_mu += [row_entity['mu'].detach().cpu()]

    XP = torch.cat(XP)
    for i, (row_M_bar, row_mu) in enumerate(zip(row_M_bars, row_mus)):
        row_M_bars[i] = torch.cat(row_M_bar)
        row_mus[i] = torch.cat(row_mu)

    for i, (col_M_bar, col_mu) in enumerate(zip(col_M_bars, col_mus)):
        col_M_bars[i] = torch.cat(col_M_bar)
        col_mus[i] = torch.cat(col_mu)

    return XP, row_M_bars, col_M_bars, row_mus, col_mus


def retrieve_embedding(net, trainloaders, norm_params, device):
    """Retrieves the learned embedding of each entity after fusion."""
    print('Retreive Embedding')
    net.eval()
    entity_embedding = {}
    for xid, trainloader in trainloaders.items():
        row_eid, col_eid = net.meta[xid]

        with torch.no_grad():
            if row_eid == col_eid:
                entity_embedding[row_eid], _ = retrieve_embedding_step(
                    net, trainloader, norm_params, device
                )
            else:
                entity_embedding[row_eid], entity_embedding[col_eid] = retrieve_embedding_step(
                    net, trainloader, norm_params, device
                )
    return entity_embedding


def retrieve_embedding_step(net, dataloaders, norm_params, device):
    """Retrieves the learned embedding of the row and column entity of a single matrix after fusion."""
    row_ids, row_loaders = dataloaders[0]
    col_ids, col_loaders = dataloaders[1]

    nrows = len(dataloaders[0][1][0])
    ncols = len(dataloaders[1][1][0])

    row_emb, col_emb = [], []
    for row_idx, mat_batches in enumerate(zip(*row_loaders)):
        Xs = [mat[0].to(device) for mat in mat_batches]
        X_bars = [Normalise(*norm_params)(X) for X in Xs]

        col_coord = (0, 0)
        for col_idx, trans_batches in enumerate(zip(*col_loaders)):
            XTs = [trans[0].to(device) for trans in trans_batches]
            XT_bars = [Normalise(*norm_params)(XT) for XT in XTs]

            _, _, row_mu, col_mu = net.emb_forward(
                X_bars, XT_bars, row_ids, col_ids)

            if row_idx == nrows - 1:  # last row
                col_emb += [col_mu.detach().cpu()]

            # print(f'{row_idx}/{nrows} | {col_idx}/{ncols}')

        row_emb += [row_mu.detach().cpu()]

    row_emb = torch.cat(row_emb)
    col_emb = torch.cat(col_emb)
    return row_emb, col_emb


def get_block(coordinates, X, XT):
    """Gets the column coordinates of the intersection of X and XT.
    
    If X.shape is (50, 100) and XT.shape is (100, 50) then the block 
    at the intersection will have the shape of (50, 50). There will 
    be 2 blocks with column coordinates of 0:50 and 50:100.
    """
    col_start, col_end = coordinates
    col_start = col_end
    col_end = col_start + XT.shape[0]

    X_block = X[:, col_start:col_end]
    coordinates = (col_start, col_end)

    return X_block, coordinates
