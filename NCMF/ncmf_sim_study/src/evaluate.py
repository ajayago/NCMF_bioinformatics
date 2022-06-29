import argparse
from src.link_prediction import *
from src.node_classification import *
from src.utils import *


def load(emb_file_path):
    emb_dict = {}
    with open(emb_file_path, 'r') as emb_file:
        for i, line in enumerate(emb_file):
            if i == 0:
                train_para = line[:-1]
            else:
                index, emb = line[:-1].split('\t')
                emb_dict[index] = np.array(emb.split()).astype(np.float32)

    return train_para, emb_dict


def record(record_file_path, train_para, scores, model, attributed, supervised):
    scores_str = dict_to_str(scores)
    with open(record_file_path, 'a') as file:
        file.write(
            f'model={model}, attributed={attributed}, supervised={supervised}\n')
        file.write(f'{train_para}\n')
        file.write(f'{scores_str}\n')
        file.write('\n')


def evaluate(data_folder, dataset, link_test_file, label_test_file, label_file, emb_file, record_file, model, task, attributed, supervised):
    print('Load Embeddings!')
    emb_file_path = f'{data_folder}/{dataset}/{emb_file}'
    train_para, emb_dict = load(emb_file_path)

    print('Start Evaluation!')
    nc_scores, lp_scores = {}, {}
    if task == 'nc' or task == 'both':
        print(
            f'Evaluate Node Classification Performance for Model {model} on Dataset {dataset}!')
        label_file_path = f'{data_folder}/{dataset}/{label_file}'
        label_test_path = f'{data_folder}/{dataset}/{label_test_file}'
        nc_scores['Macro-F1'], nc_scores['Micro-F1'] = nc_evaluate(
            dataset, supervised, label_file_path, label_test_path, emb_dict)

    if task == 'lp' or task == 'both':
        print(
            f'Evaluate Link Prediction Performance for Model {model} on Dataset {dataset}!')
        link_test_path = f'{data_folder}/{dataset}/{link_test_file}'
        lp_scores['AUC'], lp_scores['MRR'], lp_scores['recall'], lp_scores['precision'], lp_scores['f1'] = lp_evaluate(
            link_test_path, emb_dict)

    consolidated_scores = consolidate_dict(nc=nc_scores, lp=lp_scores)

    print('Record Results!')
    record_file_path = f'{data_folder}/{dataset}/{record_file}'
    record(record_file_path, train_para, consolidated_scores,
           model, attributed, supervised)

    return consolidated_scores
