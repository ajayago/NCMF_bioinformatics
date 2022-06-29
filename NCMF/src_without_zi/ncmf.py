import pandas as pd
import numpy as np
import torch
from src_without_zi.models import *
from src_without_zi.train import *
from src_without_zi.utils import *
from src_without_zi.data_utils import *
from src_without_zi.sample import *
#from src_without_zi.evaluate import *
from src_without_zi.link_prediction import *
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, mean_squared_error
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

seed = 1
max_iter = 3000
np.random.seed(seed)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

class ncmf():
    def __init__(self, sample_no, data_folder, dataset_name, matrix_types, num_epochs = 1000, learning_rate = 1e-6, weight_decay = 1e-4, convergence_threshold = -1e-3, train_batch_size = 2048, valid_batch_size = 2048, entity_matrices = ['X0', 'X1', 'X2'], pretrain = False, max_norm = 1, lamda = 1e-3, anneal = 'cosine', num_cycles = 10, proportion = 0.8, ntrain_neg = 5, nvalid_neg = 5, autoencoder_k = 50, autoencoder_k_factor = 0, autoencoder_hidden_dim = 1024, autoencoder_act_f = 'tanh', fusion_act_f = 'tanh', reconstructor_act_f = 'tanh'):
        self.node_file = f'sampled{sample_no}_node.dat'
        self.link_file = f'sampled{sample_no}_link.dat'
        self.link_test_file = f'sampled{sample_no}_link.dat.test'
        self.label_file = f'sampled{sample_no}_label.dat'
        self.label_test_file = f'sampled{sample_no}_label.dat.test'
        self.meta_file = f'sampled{sample_no}_meta.dat'
        self.info_file = f'sampled{sample_no}_info.dat'
        self.record_file = f'sampled{sample_no}_record.dat'
        self.sample_no = sample_no
        self.seed = 0
        self.cuda_id = 0
        #self.data_folder = '../../datasets/NCMF/'
        self.data_folder = data_folder
        self.matrix_types = matrix_types
        self.dataset = f'{dataset_name}'
        self.i = 0
        self.runs_folder = './runs/'
        self.emb_file = f'./emb_sample_{sample_no}.dat'
        self.writer = SummaryWriter(f'{self.runs_folder}/exp')
        set_seed(self.seed)
        self.entity_matrices = entity_matrices
        self.hyperparams = {
                'num_epochs': num_epochs, # best so far epoch 1000, wd = 1e-2, lr 1e-6, ct -1e-3, batch size 2048, activation tanh
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'convergence_threshold': convergence_threshold,
                'train_batch_size': train_batch_size,
                'valid_batch_size': valid_batch_size,
                'pretrain': pretrain,
                'max_norm': max_norm,
                'lamda': lamda,
                'anneal': anneal,
                'num_cycles': num_cycles,
                'proportion': proportion,
                'ntrain_neg': ntrain_neg,
                'nvalid_neg': nvalid_neg,
        }
        self.autoencoder_config = {
                'k': autoencoder_k,
                'k_factor': autoencoder_k_factor,
                'hidden_dim': autoencoder_hidden_dim,
                'activation_function': autoencoder_act_f,
        }
        self.fusion_config = {
                'activation_function': fusion_act_f,
        }
        self.reconstructor_config = {
                'activation_function': reconstructor_act_f,
        }

    def fit(self):
        print("NCMF without ZI Loss")
        set_seed(self.seed)
        raw_data = read_data(data_folder=self.data_folder, dataset=self.dataset, node_file=self.node_file, link_file=self.link_file, test_link_file=self.link_test_file, label_file=self.label_file, test_label_file=self.label_test_file, info_file=self.info_file, meta_file=self.meta_file)
        node_df, link_df, test_link_df, label_df, test_label_df, node_info_df, link_info_df, label_info_df, node_meta_df, link_meta_df, label_meta_df = raw_data
        graph, meta, entity_dims, node_idx_df, train_matrices, train_masks, valid_cells = load_data(raw_data, ntrain_neg=self.hyperparams['ntrain_neg'], nvalid_neg=self.hyperparams['nvalid_neg'], valid_split=0.01, seed=self.seed)
        norm_params = compute_normalisation_params(train_matrices, train_masks, binarise=True, stat='std_mean')
        trainloaders, validloaders, embloaders = load_dataloaders(graph, meta, train_matrices, train_masks, valid_cells, self.hyperparams['train_batch_size'], self.hyperparams['valid_batch_size'], emb_matrix_ids=self.entity_matrices)
        device = f'cuda:{self.cuda_id}' if torch.cuda.is_available() else 'cpu'
        net = DCMF( graph, meta, entity_dims, self.autoencoder_config, self.reconstructor_config, self.fusion_config).to(device)
        net, losses = train_and_validate( net, trainloaders, validloaders, embloaders, norm_params, self.hyperparams, device, self.writer, self.matrix_types)
        entity_embedding = retrieve_embedding( net, embloaders, norm_params, device)
        save_embedding(node_idx_df, entity_embedding, file_path=os.path.join(self.data_folder, self.dataset, self.emb_file))
        # reconstructing matrices
        self.XP, self.row_M_bar, self.col_M_bar, self.row_mu, self.col_mu = reconstruct(net, trainloaders, norm_params, device)
        for xid in self.XP.keys():
            np.save(os.path.join(self.data_folder, self.dataset) + f"/{self.sample_no}" + f"/{xid}", self.XP[f"{xid}"], allow_pickle = True)
            np.save(os.path.join(self.data_folder, self.dataset) + f"/{self.sample_no}" + f"/row_M_bar_{xid}", self.row_M_bar[f"{xid}"][0], allow_pickle = True)
            np.save(os.path.join(self.data_folder, self.dataset) + f"/{self.sample_no}" + f"/col_M_bar_{xid}", self.col_M_bar[f"{xid}"][0], allow_pickle = True)

    def load(self, emb_file_path):
        emb_dict = {}
        with open(emb_file_path, 'r') as emb_file:
            for i, line in enumerate(emb_file):
                if i == 0:
                    train_para = line[:-1]
                else:
                    index, emb = line[:-1].split('\t')
                    emb_dict[index] = np.array(emb.split()).astype(np.float32)

        return train_para, emb_dict

    def cross_validation(self, edge_embs, edge_labels):
    
        auc, mrr, recall, precision, f1, predictions, actual = [], [], [], [], [], [], []
        seed_nodes, num_nodes = np.array(list(edge_embs.keys())), len(edge_embs)
        skf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros((num_nodes,1)), np.zeros(num_nodes))):
            print(f'Start Evaluation Fold {fold}!')
            #print(train_idx)
            #print(test_idx)
            train_edge_embs, test_edge_embs, train_edge_labels, test_edge_labels = [], [], [], []
            for each in train_idx:
                train_edge_embs.append(edge_embs[seed_nodes[each]])
                train_edge_labels.append(edge_labels[seed_nodes[each]])
            for each in test_idx:
                test_edge_embs.append(edge_embs[seed_nodes[each]])
                test_edge_labels.append(edge_labels[seed_nodes[each]])
            train_edge_embs, test_edge_embs, train_edge_labels, test_edge_labels = np.concatenate(train_edge_embs), np.concatenate(test_edge_embs), np.concatenate(train_edge_labels), np.concatenate(test_edge_labels)
    
            #print(train_edge_embs)
            #print(train_edge_labels)
            clf = LinearSVC(random_state=seed, max_iter=max_iter)
            clf.fit(train_edge_embs, train_edge_labels)
            preds = clf.predict(test_edge_embs)
            predictions.extend(preds)
            actual.extend(test_edge_labels)
            auc.append(roc_auc_score(test_edge_labels, preds))
             
            # Adding precision, recall, f1 score
            recall.append(recall_score(test_edge_labels, preds))
            precision.append(precision_score(test_edge_labels, preds))
            f1.append(f1_score(test_edge_labels, preds))
    
            confidence = clf.decision_function(test_edge_embs)
            curr_mrr, conf_num = [], 0
            for each in test_idx:
                test_edge_conf = np.argsort(-confidence[conf_num:conf_num+len(edge_labels[seed_nodes[each]])])
                rank = np.empty_like(test_edge_conf)
                rank[test_edge_conf] = np.arange(len(test_edge_conf))
                try:
                    curr_mrr.append(1/(1+np.min(rank[np.argwhere(edge_labels[seed_nodes[each]]==1).flatten()])))
                except:
                    curr_mrr.append(0)
                conf_num += len(rank)
            mrr.append(np.mean(curr_mrr))
            assert conf_num==len(confidence)

        return np.mean(auc), np.mean(mrr), np.mean(recall), np.mean(precision), np.mean(f1), predictions, actual

    def record(self, record_file_path, train_para, scores, model, attributed, supervised):
        scores_str = dict_to_str(scores)
        with open(record_file_path, 'a') as file:
            file.write(
                f'model={model}, attributed={attributed}, supervised={supervised}\n')
            file.write(f'{train_para}\n')
            file.write(f'{scores_str}\n')
            file.write('\n')
        with open(f'{self.data_folder}/out/{self.dataset}_sample{self.sample_no}_NCMF_results.csv', 'w') as outfile:
            outfile.write("AUC,MRR\n")
            outfile.write(f'{scores["lp/AUC"]:.4f},{scores["lp/MRR"]:.4f}\n')
        with open(f'{self.data_folder}/out/{self.dataset}_sample{self.sample_no}_NCMF_preds.csv', 'w') as outfile:
            outfile.write("preds,actual\n")
            for i in range(len(scores["lp/predictions"])):
                outfile.write(f'{scores["lp/predictions"][i]},{scores["lp/actual"][i]}\n')
        return

    def lp_evaluate(self, test_file_path, emb_dict):
        print("Starting evaluation func") 
        posi, nega = defaultdict(set), defaultdict(set)
        with open(test_file_path, 'r') as test_file:
            for line in test_file:
                left, right, label = line[:-1].split('\t')
                if label=='1':
                    posi[left].add(right)
                elif label=='0':
                    nega[left].add(right)
        edge_embs, edge_labels = defaultdict(list), defaultdict(list)
        for left, rights in posi.items():
            for right in rights:
                edge_embs[left].append(emb_dict[left]*emb_dict[right])
                edge_labels[left].append(1)
        for left, rights in nega.items():
            for right in rights:
                edge_embs[left].append(emb_dict[left]*emb_dict[right])
                edge_labels[left].append(0)
        #print(edge_embs)
        for node in edge_embs:
            edge_embs[node] = np.array(edge_embs[node])
            edge_labels[node] = np.array(edge_labels[node])
        print("Just before cross val") 
        auc, mrr, recall, precision, f1, predictions, actual = self.cross_validation(edge_embs, edge_labels)
        return auc, mrr, recall, precision, f1, predictions, actual

    def evaluate(self):
        model = 'DataFusion'
        task = 'lp'
        attributed = 'False'
        supervised = 'False'
        emb_file_path = f'{self.data_folder}/{self.dataset}/{self.emb_file}'
        train_para, emb_dict = self.load(emb_file_path)
        nc_scores, lp_scores = {}, {}
        link_test_path = f'{self.data_folder}/{self.dataset}/{self.link_test_file}'
        print("Start eval")
        lp_scores['AUC'], lp_scores['MRR'], lp_scores['recall'], lp_scores['precision'], lp_scores['f1'], lp_scores['predictions'], lp_scores['actual'] = self.lp_evaluate(link_test_path, emb_dict)

        consolidated_scores = consolidate_dict(nc=nc_scores, lp=lp_scores)

        print('Record Results!')
        record_file_path = f'{self.data_folder}/{self.dataset}/{self.record_file}'
        self.record(record_file_path, train_para, consolidated_scores, model, attributed, supervised)

#        scores = evaluate(self.data_folder, self.dataset, self.link_test_file, self.label_test_file, self.label_file, self.emb_file, self.record_file, model, task, attributed, supervised)
        dict_params = {"hyperparameter_config": self.hyperparams,
                "autoencoder_config": self.autoencoder_config,
                "reconstructor_config": self.reconstructor_config,
                "fusion_config": self.fusion_config}
        self.out_dict_info = {"params": dict_params,
                "auc": lp_scores['AUC'],
                "mrr": lp_scores['MRR'],
                "recall": lp_scores['recall'],
                "precision": lp_scores['precision'],
                "F1": lp_scores['f1']
                }
        print("NCMF eval done") 

    def get_rmse(self, Rpred, Rtest, matrix_part = "ones"):
        print("Computing RMSE.")
        print("Rpred.shape: ",Rpred.shape)
        print("Rtest.shape: ",Rtest.shape)
        if matrix_part == "ones":
            row_idx_list,col_idx_list = Rtest.nonzero()
        else:
            row_idx_list, col_idx_list = (Rtest==0).nonzero() # getting all zeros from the test matrix
        nz_list_pred = []
        nz_list_test = []
        for i in np.arange(len(row_idx_list)):
            r = row_idx_list[i]
            c = col_idx_list[i]
            nz_list_pred.append(Rpred[r,c])
            nz_list_test.append(Rtest[r,c])
        print("nz_list_pred.shape: ",len(nz_list_pred))
        print("nz_list_test.shape: ",len(nz_list_test))
        rmse = np.sqrt(mean_squared_error(nz_list_test,nz_list_pred))
        ##mse = np.sum(np.absolute(nz_list_test-nz_list_pred))
        return rmse

    def get_rmse_real_matrix(self, Rpred, Rtest, idx):
        # idx is the list of indices which need to be compared between Rpred and Rtest
        # total_idx_list = set(range(0, Rpred.shape[0] * Rpred.shape[1]))
        Rpred_flattened = Rpred.flatten()
        Rtest_flattened = Rtest.flatten()
        nz_list_pred = []
        nz_list_test = []
        for i in idx:
            nz_list_pred.append(Rpred_flattened[i])
            nz_list_test.append(Rtest_flattened[i])
        print("nz_list_pred.shape: ",len(nz_list_pred))
        print("nz_list_test.shape: ",len(nz_list_test))
        rmse = np.sqrt(mean_squared_error(nz_list_test,nz_list_pred))
        return rmse
       
    def get_auc(self, Rpred, Rtest, idx):
        print("Computing AUC")
        Rpred_flattened = Rpred.flatten()
        Rtest_flattened = Rtest.flatten()
        nz_list_pred = []
        nz_list_test = []
        for i in idx:
            nz_list_pred.append(Rpred_flattened[i])
            nz_list_test.append(Rtest_flattened[i])
        print("nz_list_pred.shape: ",len(nz_list_pred))
        print("nz_list_test.shape: ",len(nz_list_test))
        auc = roc_auc_score(nz_list_test, nz_list_pred)
        return auc

    def evaluate_rmse(self, pred_matrix, path_to_truth_matrix, test_indices_file = None, matrix_part = "ones", istrainRMSE = False):
        #Rpred = np.load(os.path.join(self.data_folder, self.dataset) + f"/{pred_matrix}.npy")
        Rpred = np.load(pred_matrix)
        if test_indices_file:
            indices_df = pd.read_csv(test_indices_file)
            indices_df.columns = ["indices"]
            test_idx = list(indices_df["indices"])
            total_idx_set = set(range(0, Rpred.shape[0] * Rpred.shape[1]))
            train_idx = list(total_idx_set - set(test_idx))
            if istrainRMSE:
                idx = train_idx
            else:
                idx = test_idx
        else:
            idx = list(range(0, Rpred.shape[0] * Rpred.shape[1]))
        if "csv" in path_to_truth_matrix:
            test_data_np = pd.read_csv(path_to_truth_matrix, header = None).values
        elif "pkl" in path_to_test_matrix:
            test_data_np = pd.read_pickle(path_to_truth_matrix)
        else:
            print("Invalid format for test matrix")
            exit()
        Rtest = test_data_np
        if matrix_part == "full":
            rmse = self.get_rmse_real_matrix(Rpred, Rtest, idx)
            print(f"Full RMSE = {rmse}")
        elif matrix_part == "ones":
            rmse = self.get_rmse(Rpred, Rtest, matrix_part = "ones")
            print(f"Ones RMSE = {rmse}")
        elif matrix_part == "zeros":
            rmse = self.get_rmse(Rpred, Rtest, matrix_part = "zeros")
            print(f"Zeros RMSE = {rmse}")
        elif matrix_part == "all":
            rmse_ones = self.get_rmse(Rpred, Rtest, matrix_part = "ones")
            rmse_full = self.get_rmse_real_matrix(Rpred, Rtest, idx)
            rmse = self.get_rmse(Rpred, Rtest, matrix_part = "zeros")
            print(f"Full RMSE = {rmse_full}")
            print(f"Ones RMSE = {rmse_ones}")
            print(f"Zeros RMSE = {rmse}")
        else:
            print("Not a valid entry for matrix_part")
            exit()

    def evaluate_auc(self, pred_matrix, path_to_truth_matrix, test_indices_file = None, istrainRMSE = False):
        Rpred = np.load(pred_matrix)
        if test_indices_file:
            indices_df = pd.read_csv(test_indices_file)
            indices_df.columns = ["indices"]
            test_idx = list(indices_df["indices"])
            total_idx_set = set(range(0, Rpred.shape[0] * Rpred.shape[1]))
            train_idx = list(total_idx_set - set(test_idx))
            if istrainRMSE:
                idx = train_idx
            else:
                idx = test_idx
        else:
            idx = list(range(0, Rpred.shape[0] * Rpred.shape[1]))
        if "csv" in path_to_truth_matrix:
            test_data_np = pd.read_csv(path_to_truth_matrix, header = None).values
        elif "pkl" in path_to_test_matrix:
            test_data_np = pd.read_pickle(path_to_truth_matrix)
        else:
            print("Invalid format for test matrix")
            exit()
        Rtest = test_data_np
        auc = self.get_auc(Rpred, Rtest, idx)
        print(f"AUC = {auc}")
