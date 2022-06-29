from src_without_decoder.models import *
from src_without_decoder.train import *
from src_without_decoder.data_utils import *
from src_without_decoder.utils import set_seed
from src_without_decoder.evaluate import evaluate
from torch.utils.tensorboard import SummaryWriter


seed = 0
set_seed(seed)

i = 0
cuda_id = 0
data_folder = 'datasets'
dataset = 'PubMed'
runs_folder = './runs/'

node_file = 'sampled1_node.dat'
link_file = 'sampled1_link.dat'
link_test_file = 'sampled1_link.dat.test'
label_file = 'sampled1_label.dat'
label_test_file = 'sampled1_label.dat.test'
meta_file = 'sampled1_meta.dat'
info_file = 'sampled1_info.dat'

raw_data = read_data(
    data_folder=data_folder,
    dataset=dataset,
    node_file=node_file,
    link_file=link_file,
    test_link_file=link_test_file,
    label_file=label_file,
    test_label_file=label_test_file,
    info_file=info_file,
    meta_file=meta_file,
)
device = f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu'

ntrain_negs = [5]
nvalid_negs = [5]
lamdas = [1e-3, 0.1, 0.3, 0.5]
weight_decays = [1e-4]
learning_rates = [1e-4]
hidden_dims = [1024, 512, 256, 128]

for ntrain_neg, nvalid_neg in zip(ntrain_negs, nvalid_negs):
    for lr in learning_rates:
        for lamda in lamdas:
            emb_file = f'emb_{i}.dat'
            writer = SummaryWriter(f'{runs_folder}/exp_{i}')

            hyperparams = {
                'num_epochs': 10,
                'learning_rate': lr,
                'weight_decay': 1e-4,
                'convergence_threshold': 1e-5,
                'train_batch_size': 2048,
                'valid_batch_size': 2048,
                'pretrain': False,
                'max_norm': 1,
                'lamda': lamda,
                'anneal': 'cosine',
                'num_cycles': 10,
                'proportion': 0.8,
                'ntrain_neg': ntrain_neg,
                'nvalid_neg': nvalid_neg,
            }
            autoencoder_config = {
                'k': 50,
                'k_factor': 0,
                'hidden_dim': 1024,
                'activation_function': 'relu',
            }
            fusion_config = {
                'activation_function': 'relu',
            }
            reconstructor_config = {
                'activation_function': 'relu',
            }

            graph, meta, entity_dims, node_idx_df, train_matrices, train_masks, valid_cells = load_data(
                raw_data,
                ntrain_neg=hyperparams['ntrain_neg'],
                nvalid_neg=hyperparams['nvalid_neg'],
                valid_split=0.01,
                seed=seed
            )
            norm_params = compute_normalisation_params(
                train_matrices, train_masks, binarise=True, stat='std_mean')
            # TODO: PubMed: emb_matrix_ids=['X1', 'X6']
            # TODO: EmbMNIST: emb_matrix_ids=['X0']
            trainloaders, validloaders, embloaders = load_dataloaders(
                graph, meta,
                train_matrices, train_masks, valid_cells,
                hyperparams['train_batch_size'], hyperparams['valid_batch_size'],
                emb_matrix_ids=['X1', 'X6']
            )

            net = DCMF(
                graph, meta, entity_dims,
                autoencoder_config, reconstructor_config, fusion_config
            ).to(device)
            net, _ = train_and_validate(
                net,
                trainloaders, validloaders, embloaders,
                norm_params, hyperparams,
                device,
                writer
            )
            entity_embedding = retrieve_embedding(
                net,
                embloaders,
                norm_params,
                device
            )
            save_embedding(
                node_idx_df,
                entity_embedding,
                file_path=os.path.join(data_folder, dataset, emb_file)
            )

            model = 'DataFusion'
            task = 'both'
            attributed = 'False'
            supervised = 'False'
            record_file = 'record.dat'

            try:
                if dataset == 'Mashup':
                    raw_data_folder = '../../../data/lianyh/Mashup_Dataset/'
                    scores = evaluate_mashup(
                        raw_data_folder, data_folder,
                        dataset, model, emb_file
                    )
                else:
                    scores = evaluate(
                        data_folder, dataset,
                        link_test_file, label_test_file, label_file,
                        emb_file, record_file, model,
                        task, attributed, supervised
                    )

            except ValueError:
                print('NAN')
                scores = {}

            hparam_dict = consolidate_dict(
                hparam=hyperparams,
                aec=autoencoder_config,
                fus=fusion_config,
                rec=reconstructor_config,
            )
            metric_dict = scores
            writer.add_hparams(hparam_dict, metric_dict)
            i += 1
