import warnings
import numpy as np
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning


seed = 1
max_iter = 3000
np.random.seed(seed)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def cross_validation(edge_embs, edge_labels):
    
    auc, mrr, recall, precision, f1 = [], [], [], [], []
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
        #print(train_edge_embs.shape)
        clf = LinearSVC(random_state=seed, max_iter=max_iter)
        clf.fit(train_edge_embs, train_edge_labels)
        preds = clf.predict(test_edge_embs)
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
        
    return np.mean(auc), np.mean(mrr), np.mean(recall), np.mean(precision), np.mean(f1)


def lp_evaluate(test_file_path, emb_dict):
    
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
            
    for node in edge_embs:
        edge_embs[node] = np.array(edge_embs[node])
        edge_labels[node] = np.array(edge_labels[node])
    
    auc, mrr, recall, precision, f1 = cross_validation(edge_embs, edge_labels)
    return auc, mrr, recall, precision, f1
    
