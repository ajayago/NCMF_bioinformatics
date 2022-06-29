import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import math
from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.model_selection import train_test_split
from src_without_zi.datasets import *
from shutil import copyfile
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"]="1"
def read_nodes(root, dataset_dir, node_file):
    node_df = pd.read_csv(
        os.path.join(root, dataset_dir, node_file),
        sep='\t',
        header=None,
        names=['node_id', 'node_name', 'node_type', 'node_attributes'],
        quoting=3  # QUOTE_NONE
    )
    return node_df


def read_links(root, dataset_dir, link_file):
    link_df = pd.read_csv(
        os.path.join(root, dataset_dir, link_file),
        sep='\t',
        header=None, names=['node_id_from', 'node_id_to', 'link_type', 'link_weight']
    )
    return link_df


def read_test_links(data_folder, dataset, file):
    link_test_df = pd.read_csv(
        os.path.join(data_folder, dataset, file),
        sep='\t',
        header=None,
        names=['node_id_from', 'node_id_to', 'link_status']
    )
    return link_test_df


def read_labels(data_folder, dataset, file):
    label_df = pd.read_csv(
        os.path.join(data_folder, dataset, file),
        sep='\t',
        header=None,
        names=['node_id', 'node_name', 'node_type', 'node_label']
    )
    return label_df


def read_test_labels(data_folder, dataset, file):
    test_label_df = pd.read_csv(
        os.path.join(data_folder, dataset, file),
        sep='\t',
        header=None,
        names=['node_id', 'node_name', 'node_type', 'node_label']
    )
    return test_label_df


def read_meta(data_folder, dataset, file):
    node_type_count = {}
    edge_type_count = {}
    label_type_count = {}
    with open(os.path.join(data_folder, dataset, file), 'r') as file:
        for line in file:
            entity, info, _, count = line.split(' ')
            info = info[:-1].split('_')
            if entity == 'Node' and info[0] == 'Type':
                node_type_count[int(info[1])] = int(count)
            if entity == 'Edge' and info[0] == 'Type':
                edge_type_count[int(info[1])] = int(count)
            if entity == 'Label' and info[0] == 'Class' and info[2] == 'Type':
                label_type_count[int(info[3])] = int(count)
    return node_type_count, edge_type_count, label_type_count


def read_info(root, dataset_dir, node_file, link_file, label_file, info_file):
    node_dict_info, link_dict_info, label_dict_info = defaultdict(
        list), defaultdict(list), defaultdict(list)
    flag = 0
    is_header = True
    with open(os.path.join(root, dataset_dir, info_file), 'r') as file:
        for line in file:
            if line.strip() == 'node.dat':
                # print('Node Info')
                flag = 1
                is_header = True
            if line.strip() == 'link.dat':
                # print('Link Info')
                flag = 2
                is_header = True
            if line.strip() == 'label.dat':
                # print('Label Info')
                flag = 3
                is_header = True

            split_line = line.split('\t')
            if len(split_line) > 1:
                if flag == 1:
                    if is_header:
                        node_type_hdr, node_meaning_hdr = line.split()
                        is_header = False
                    else:
                        node_type, node_meaning = line.split()
                        node_dict_info[node_type_hdr].append(int(node_type))
                        node_dict_info[node_meaning_hdr].append(node_meaning)
                if flag == 2:
                    if is_header:
                        link_hdr, link_start_hdr, link_end_hdr, link_meaning_hdr = line.split()
                        is_header = False
                    else:
                        link, link_start, link_end, link_meaning = line.split()
                        link_dict_info[link_hdr].append(int(link))
                        link_dict_info[link_start_hdr].append(int(link_start))
                        link_dict_info[link_end_hdr].append(int(link_end))
                        link_dict_info[link_meaning_hdr].append(link_meaning)
                if flag == 3:
                    if is_header:
                        label_type_hdr, label_class_hdr, label_meaning_hdr = line.split()
                        is_header = False
                    else:
                        label_type, label_class, label_meaning = line.split()
                        label_dict_info[label_type_hdr].append(int(label_type))
                        label_dict_info[label_class_hdr].append(
                            int(label_class))
                        label_dict_info[label_meaning_hdr].append(
                            label_meaning)

    node_info_df = pd.DataFrame(node_dict_info)
    link_info_df = pd.DataFrame(link_dict_info)
    label_info_df = pd.DataFrame(label_dict_info)

    return node_info_df, link_info_df, label_info_df


def combine_meta_info(info_df, type_df, catergory):
    if catergory == 'node':
        left_on = 'TYPE'
        right_on = 'node_type'
    if catergory == 'link':
        left_on = 'LINK'
        right_on = 'edge_type'
    if catergory == 'label':
        left_on = 'CLASS'
        right_on = 'label_type'

    meta_info = pd.merge(info_df, type_df, how='left',
                         left_on=left_on, right_on=right_on)
    meta_info = meta_info.drop([right_on], axis=1)
    meta_info.columns = map(str.lower, meta_info.columns)
    return meta_info


def read_data(**kwargs):
    node_df = read_nodes(
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['node_file']
    )
    link_df = read_links(
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['link_file']
    )
    test_link_df = read_test_links(
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['test_link_file']
    )
    label_df = read_labels(
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['label_file']
    )
    test_label_df = read_test_labels(
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['test_label_file']
    )
    node_info_df, link_info_df, label_info_df = read_info(
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['node_file'],
        kwargs['link_file'],
        kwargs['label_file'],
        kwargs['info_file']
    )
    node_meta_df, link_meta_df, label_meta_df = read_meta(
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['meta_file'],
    )
    return node_df, link_df, test_link_df, label_df, test_label_df, node_info_df, link_info_df, label_info_df, node_meta_df, link_meta_df, label_meta_df


def write_df(df, data_folder, dataset, file):
    df.to_csv(
        path_or_buf=os.path.join(data_folder, dataset, file),
        sep='\t',
        header=False,
        quoting=3,  # QUOTE_NONE
        index=False
    )


def write_meta(node_df, link_df, test_link_df, label_df, test_label_df, data_folder, dataset, file):
    node_count_df = node_df.groupby('node_type').size().to_frame()
    node_count_df.columns = ['count']
    node_count_df.reset_index(level=0, inplace=True)

    tmp_link_df = link_df.drop(['link_weight'], axis=1)
    tmp_test_link_df = test_link_df[test_link_df['link_status'] == 1]
    tmp_test_link_df = tmp_test_link_df.drop(['link_status'], axis=1)
    # TODO: disease-disease link type 2, gene-disease link type = 1
    tmp_test_link_df['link_type'] = 1
    comb_link_df = pd.concat([tmp_link_df, tmp_test_link_df])

    link_count_df = comb_link_df.groupby('link_type').size().to_frame()
    link_count_df.columns = ['count']
    link_count_df.reset_index(level=0, inplace=True)

    comb_link_df = pd.concat([label_df, test_label_df])
    label_count_df = comb_link_df.groupby('node_label').size().to_frame()
    label_count_df.columns = ['count']
    label_count_df.reset_index(level=0, inplace=True)

    write_path = os.path.join(data_folder, dataset, file)
    with open(write_path, 'w') as file:
        num_total = node_count_df['count'].sum()
        file.write(f'Node Total: Count {num_total}\n')
        for (type, count) in node_count_df.itertuples(index=False):
            file.write(f'Node Type_{type}: Count {count}\n')

        num_total = link_count_df['count'].sum()
        file.write(f'Edge Total: Count {num_total}\n')
        for (type, count) in link_count_df.itertuples(index=False):
            file.write(f'Edge Type_{type}: Count {count}\n')

        num_total = label_count_df['count'].sum()
        file.write(f'Label Total: Count {num_total}\n')
        label_class = 1
        num_class_total = num_total
        file.write(
            f'Label Class_{label_class}_Total: Count {num_class_total}\n')
        for (type, count) in label_count_df.itertuples(index=False):
            file.write(
                f'Label Class_{label_class}_Type_{type}: Count {count}\n')


# TODO: write info instead of copy
def write_info(data_folder, dataset, file):
    copyfile(
        src=os.path.join(data_folder, dataset, 'info.dat'),
        dst=os.path.join(data_folder, dataset, file)
    )


def write_data(data, **kwargs):
    node_df, link_df, test_link_df, label_df, test_label_df, _, _, _, _, _, _ = data

    write_df(
        node_df,
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['node_file']
    )
    write_df(
        link_df,
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['link_file']
    )
    write_df(
        test_link_df,
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['test_link_file']
    )
    write_df(
        label_df,
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['label_file']
    )
    write_df(
        test_label_df,
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['test_label_file']
    )
    write_meta(
        node_df,
        link_df, test_link_df,
        label_df, test_label_df,
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['meta_file']
    )
    write_info(
        kwargs['data_folder'],
        kwargs['dataset'],
        kwargs['info_file']
    )


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if isinstance(sample, sp.sparse.csr.csr_matrix):
            return torch.from_numpy(sample.toarray())

        return torch.from_numpy(sample)


class ToSparseTensor(object):
    """Convert ndarrays in sample to sparse Tensors."""

    def __call__(self, sample):
        coo_mat = sample.tocoo()
        indices = torch.LongTensor([coo_mat.row, coo_mat.col])
        values = torch.FloatTensor(coo_mat.data)

        return torch.sparse.FloatTensor(indices, values, torch.Size(coo_mat.shape))


class Binarise(object):
    """Binarise a tensor."""

    def __call__(self, sample):
        return torch.where(sample > 0, torch.tensor(1), torch.tensor(0)).type(torch.float32)

    def __repr__(self):
        return self.__class__.__name__


class Normalise(object):
    """Normalises a tensor."""

    def __init__(self, std, mean):
        self.std = std
        self.mean = mean

    def __call__(self, sample):
        sample = torch.log1p(sample.type(torch.float32))
        sample = (sample - self.mean) / self.std
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(std={}, mean={})'.format(self.std, self.mean)


def compute_normalisation_params(matrices, masks, binarise=False, stat='std_mean'):
    weights = np.concatenate(
        [mat[mask] for mat, mask in zip(matrices.values(), masks.values())]
    )
    if binarise:
        weights = np.where(weights > 0, np.array([1]), np.array(0))

    if stat == 'std_mean':
        return weights.std(), weights.mean()
    elif stat == 'min_max':
        return weights.min(), weights.max()
    else:
        raise RuntimeError(
            f'{stat} unavailable. Choose one of std_mean, min_max')


def sample_link_zeros(node_df, link_info_df, link_ones_df, full_links, k=5, seed=0):
    link_ones_from_count_df = link_ones_df[['link_type', 'node_id_from']].groupby(
        ['link_type', 'node_id_from']).size().reset_index(name='count')
    link_zeros_rows, link_zeros_cols, link_zeros_vals, link_zeros_typs = [], [], [], []
    for link_type, node_from_id, node_from_count in link_ones_from_count_df.itertuples(index=False, name=None):
        node_to_id_zeros = np.where(full_links[node_from_id, :] < 1)[
            0]  # 0 values are negative links in train set
        # entity type of column
        node_to_type = link_info_df[link_info_df['LINK']
                                    == link_type]['END'].values[0]

        # subset ids of specified link type
        filtered_node_ids = node_df[node_df['node_type']
                                    == node_to_type][['node_id']]
        
        filtered_node_ids = filtered_node_ids[filtered_node_ids.isin(node_to_id_zeros)[
            'node_id'].values]
        
        # randomly pick k ids as negative links (zeroes)
        num_neg_links = k * node_from_count
        if num_neg_links > len(filtered_node_ids['node_id']):
            num_neg_links = len(filtered_node_ids['node_id'])

        sample_link_zeros_cols = filtered_node_ids['node_id'].sample(
            n=num_neg_links, random_state=seed).tolist()
        link_zeros_cols += sample_link_zeros_cols
        link_zeros_rows += num_neg_links * [node_from_id]
        link_zeros_vals += num_neg_links * [0]
        link_zeros_typs += num_neg_links * [link_type]

    link_zeros_df = pd.DataFrame(data={
        'node_id_from': link_zeros_rows,
        'node_id_to': link_zeros_cols,
        'link_type': link_zeros_typs,
        'link_weight': link_zeros_vals,
    })
    return link_zeros_df


def train_valid_split(node_df, link_df, test_link_df, link_info_df, ntrain_neg=5, nvalid_neg=5, valid_size=0.01, seed=0):
    """Splits links (positve and negative) into train and validation set.

    The positive links in the validation set are obtained by splitting
    the train set by the specified validation size. The negative links are
    obtained by sampling the same number of positive links in the validation set
    for each node from the train set (test set links are excluded from sample).
    """
    # matrix
    # train: 0 (negative link), 1 (positive link)
    # valid: 2 (negative link), 3 (positive link)
    # test: 4 (negative link), 5 (positive link)
    full_links = np.zeros((node_df.shape[0], node_df.shape[0]), dtype=np.int8)

    # include test links
    test_link_ones_rows = test_link_df[test_link_df['link_status'] == 1][[
        'node_id_from']].values
    print(len(test_link_ones_rows))
    test_link_ones_cols = test_link_df[test_link_df['link_status'] == 1][[
        'node_id_to']].values
    if len(test_link_ones_rows)>0 and len(test_link_ones_cols)>0:
        full_links[test_link_ones_rows, test_link_ones_cols] = 5

    test_link_zeros_rows = test_link_df[test_link_df['link_status'] == 0][[
        'node_id_from']].values
    test_link_zeros_cols = test_link_df[test_link_df['link_status'] == 0][[
        'node_id_to']].values
    if len(test_link_zeros_rows)>0 and len(test_link_zeros_cols)>0:
        full_links[test_link_zeros_rows, test_link_zeros_cols] = 4

    # split positive links
    train_link_ones_df, valid_link_ones_df = train_test_split(
        link_df, test_size=valid_size, random_state=seed)

    # include positive train links
    train_link_ones_rows = train_link_ones_df[['node_id_from']].values
    train_link_ones_cols = train_link_ones_df[['node_id_to']].values
    full_links[train_link_ones_rows, train_link_ones_cols] = 1

    # include positive validation links
    valid_link_ones_rows = valid_link_ones_df[['node_id_from']].values
    valid_link_ones_cols = valid_link_ones_df[['node_id_to']].values
    full_links[valid_link_ones_rows, valid_link_ones_cols] = 3

    # split negative links
    valid_link_zeros_df = sample_link_zeros(
        node_df, link_info_df, valid_link_ones_df, full_links, k=nvalid_neg, seed=seed)
    valid_link_df = pd.concat([valid_link_ones_df, valid_link_zeros_df])

    # include negative validation links
    valid_link_zeros_rows = valid_link_zeros_df[['node_id_from']].values
    valid_link_zeros_cols = valid_link_zeros_df[['node_id_to']].values
    full_links[valid_link_zeros_rows, valid_link_zeros_cols] = 2

    # include negative train links
    train_link_zeros_df = sample_link_zeros(
        node_df, link_info_df, train_link_ones_df, full_links, k=ntrain_neg, seed=seed)
    train_link_df = pd.concat([train_link_ones_df, train_link_zeros_df])

    return train_link_df, valid_link_df


def assign_link_type(df, node_df, link_info_df):
    merged_df = pd.merge(df, node_df[['node_id', 'node_type']],
                         how='left', left_on='node_id_from', right_on='node_id')
    merged_df.rename(columns={'node_type': 'node_type_from'}, inplace=True)
    merged_df.drop('node_id', axis=1, inplace=True)

    merged_df = pd.merge(merged_df, node_df[[
        'node_id', 'node_type']], how='left', left_on='node_id_to', right_on='node_id')
    merged_df.rename(columns={'node_type': 'node_type_to'}, inplace=True)
    merged_df.drop('node_id', axis=1, inplace=True)

    for _, _, _, node_type_from, node_type_to in merged_df.itertuples(index=False, name=None):
        link_type = link_info_df[(link_info_df['START'] == node_type_from) & (
            link_info_df['END'] == node_type_to)]['LINK'].values[0]
        merged_df['link_type'] = link_type

    merged_df.drop(['node_type_from', 'node_type_to'], axis=1, inplace=True)
    return merged_df


def init_id_idx_map(node_df):
    """Maps a matrix index to each node id from a specific node type."""
    num_node_types = node_df.groupby(['node_type']).size().shape[0]
    node_idx_dfs = []
    for node_type in range(num_node_types):
        num_nodes = node_df[node_df['node_type'] == node_type].shape[0]
        temp_df = node_df[node_df['node_type'] == node_type].assign(
            idx=[i for i in range(num_nodes)])
        node_idx_dfs.append(temp_df)
    return pd.concat(node_idx_dfs)


def map_node_id_to_matrix_idx(df, node_idx_df):
    """Maps the node ids of the specified dataframe with using the mappings given."""
    merged_df = pd.merge(df, node_idx_df[[
        'node_id', 'idx']], how='left', left_on='node_id_from', right_on='node_id')
    merged_df.rename(columns={'idx': 'idx_from'}, inplace=True)
    merged_df.drop('node_id', axis=1, inplace=True)

    merged_df = pd.merge(merged_df, node_idx_df[[
        'node_id', 'idx']], how='left', left_on='node_id_to', right_on='node_id')
    merged_df.rename(columns={'idx': 'idx_to'}, inplace=True)
    merged_df.drop('node_id', axis=1, inplace=True)
    return merged_df


def init_matrix(link_df, shape):
    vals = link_df['link_weight'].to_numpy()
    rows = link_df['idx_from'].to_numpy()
    cols = link_df['idx_to'].to_numpy()
    matrix = coo_matrix((vals, (rows, cols)), shape).toarray()
    return matrix


def init_mask(link_df, shape):
    mask_rows = link_df['idx_from'].to_numpy()
    mask_cols = link_df['idx_to'].to_numpy()

    mask = np.zeros(shape)
    mask[mask_rows, mask_cols] = 1.
    mask = mask > 0
    return mask


def init_matrices_and_mask(link_df, link_info_df, entity_dims):
    matrices, masks = {}, {}
    for link_type, node_start, node_end, _ in link_info_df.itertuples(index=False):
        id = f'X{link_type}'
        shape = (
            entity_dims['e' + str(node_start)],
            entity_dims['e' + str(node_end)]
        )

        links = link_df[link_df['link_type'] == link_type]
        matrices[id] = init_matrix(links, shape)
        masks[id] = init_mask(links, shape)

    return matrices, masks


def init_cells(link_df, link_info_df):
    cells = {}
    for link_type, _, _, _ in link_info_df.itertuples(index=False):
        id = f'X{link_type}'
        links = link_df[link_df['link_type'] == link_type]
        cells[id] = (links['idx_from'].to_numpy(),
                     links['idx_to'].to_numpy(), links['link_weight'].to_numpy())
    return cells


def init_dataset(matrix, mask, interaction, train_batch_size, transpose):
    transform = torchvision.transforms.Compose(
        [Binarise()]
    )
    # transform = None
    trainset = MatrixDatasetV2(
        matrix=matrix if not transpose else matrix.T,
        mask=mask if not transpose else mask.T,
        batch_size=train_batch_size,
        transform=transform,
        mask_transform=None
    )
    validset = CellInteractionDataset(
        rows=interaction[0],
        columns=interaction[1],
        values=interaction[2],
        transform=None,
        target_transform=transform
    )
    return trainset, validset


def load_graph(node_info_df, link_info_df):
    graph = {}
    for entity_type, _ in node_info_df.itertuples(index=False):
        matrices = []
        for link_type, start, end, _ in link_info_df.itertuples(index=False):
            if entity_type == start or entity_type == end:
                matrices.append(f'X{link_type}')

        graph[f'e{entity_type}'] = matrices
    return graph


def load_meta(link_info_df):
    meta = {
        f'X{link_type}': (f'e{start}', f'e{end}')
        for link_type, start, end, _ in link_info_df.itertuples(index=False)
    }
    return meta


def load_entity_dims(node_meta_df):
    entity_dims = {f'e{eid}': dim for eid, dim in node_meta_df.items()}
    return entity_dims


# TODO: Create algo to determine least number of matrices with all entities
def load_embloaders(dataloaders, ids):
    embloaders = {id: dataloaders[id] for id in ids}
    return embloaders


def load_data(raw_data, ntrain_neg=5, nvalid_neg=5, valid_split=0.01, seed=0):
    node_df, link_df, test_link_df, _, _, node_info_df, link_info_df, _, node_meta_df, _, _ = raw_data

    graph = load_graph(node_info_df, link_info_df)
    meta = load_meta(link_info_df)
    entity_dims = load_entity_dims(node_meta_df)

    print('Mapping node ids to matrix indices...')
    node_idx_df = init_id_idx_map(node_df)

    print('Splitting training and validation links...')
    train_link_df, valid_link_df = train_valid_split(
        node_df, link_df, test_link_df, link_info_df,
        ntrain_neg, nvalid_neg, valid_split, seed=seed
    )
    train_link_df = map_node_id_to_matrix_idx(train_link_df, node_idx_df)
    valid_link_df = map_node_id_to_matrix_idx(valid_link_df, node_idx_df)

    print('Loading matrices and masks...')
    train_matrices, train_masks = init_matrices_and_mask(
        train_link_df, link_info_df, entity_dims
    )
    valid_cells = init_cells(valid_link_df, link_info_df)
    return graph, meta, entity_dims, node_idx_df, train_matrices, train_masks, valid_cells


def load_dataloaders(graph, meta, train_matrices, train_masks, valid_cells, train_batch_size, valid_batch_size, emb_matrix_ids):
    train_data, validloaders = {}, {}
    for id in train_matrices.keys():
        matrix = train_matrices[id]
        mask = train_masks[id]
        interaction = valid_cells[id]

        trainloader, validloader = load_dataloader(
            datasets=init_dataset(
                matrix,
                mask,
                interaction,
                train_batch_size,
                transpose=False
            ),
            valid_batch_size=valid_batch_size,
        )
        trainloader_t, _ = load_dataloader(
            datasets=init_dataset(
                matrix,
                mask,
                interaction,
                train_batch_size,
                transpose=True
            ),
            valid_batch_size=valid_batch_size,
        )
        train_data[id] = (trainloader, trainloader_t)
        validloaders[id] = validloader

    trainloaders = gather_dataloaders(graph, meta, train_data)
    embloaders = load_embloaders(trainloaders, emb_matrix_ids)
    return trainloaders, validloaders, embloaders


def load_dataloader(**kwargs):
    trainset, validset = kwargs['datasets']
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        collate_fn=None
    )
    validloader = torch.utils.data.DataLoader(
        dataset=validset,
        batch_size=kwargs['valid_batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=None
    )

    return trainloader, validloader


def gather_dataloaders(graph, meta, data):
    group_data = {}
    for id, entities in meta.items():
        print(f'To reconstruct {id}')
        dim_data = []
        for dim, entity in enumerate(entities):
            ids, loaders = [], []
            print(f'dim:{dim}; {entity}')
            for dim_id in graph[entity]:
                if entity == meta[dim_id][0]:
                    print(dim_id, entity, 'row')
                    if id == dim_id:
                        ids.insert(0, (dim_id, 'row'))
                        loaders.insert(0, data[dim_id][0])
                    else:
                        ids.append((dim_id, 'row'))
                        loaders.append(data[dim_id][0])
                if entity == meta[dim_id][1]:
                    print(dim_id, entity, 'col')
                    if id == dim_id:
                        ids.insert(0, (dim_id, 'col'))
                        loaders.insert(0, data[dim_id][1])
                    else:
                        ids.append((dim_id, 'col'))
                        loaders.append(data[dim_id][1])
            dim_data.append((ids, loaders))
        group_data[id] = tuple(dim_data)
    return group_data


def subset_rel_trainloaders(ids, dataloaders):
    return {id: dataloaders[id] for id in ids}
