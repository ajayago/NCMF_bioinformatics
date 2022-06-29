import torch
import numpy as np
import pandas as pd
import random
from collections import MutableMapping


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clip(x, min_x):
    device = x.get_device()
    min_x = torch.tensor([min_x]).to(device)
    clipped_x = torch.min(x, min_x)
    return clipped_x


def convert_values(matrix, from_val, to_val):
    matrix[matrix == from_val] = to_val
    return matrix


def save_embedding(node_idx_df, entity_embedding, file_path='./emb.dat'):
    """Writes the given embeddings to file with its corresponding node id."""
    entity_embedding_df = {id: pd.DataFrame(
        emb.numpy()) for id, emb in entity_embedding.items()}
    node_types = node_idx_df['node_type'].value_counts(
    ).sort_index().index.to_list()

    entity_embedding_id_idx = {}
    for node_type in node_types:
        entity_map = pd.merge(node_idx_df[node_idx_df['node_type'] == node_type],
                              entity_embedding_df[f'e{node_type}'], left_on='idx', right_index=True)
        entity_map = entity_map.drop(
            columns=['node_name', 'node_type', 'node_attributes'])
        entity_map = entity_map.reset_index(drop=True)

        entity_embedding_id_idx[f'e{node_type}'] = entity_map

    embedding_df = pd.concat([df for df in entity_embedding_id_idx.values()])
    embedding_df = embedding_df.sort_values(by=['node_id'])
    embedding_df = embedding_df.drop(columns=['idx'])
    write_to_file('', embedding_df, file_path)


def write_to_file(params, emb_df, write_path):
    """Writes training parameters and embedding dataframe to file."""
    with open(write_path, 'w') as file:
        file.write(f'{params}\n')
        for idx, row in emb_df.iterrows():
            id = int(row['node_id'])
            emb = row[1:].astype(np.float32)
            emb_str = ' '.join(emb.astype(str))
            file.write(f'{id}\t{emb_str}\n')


def consolidate_dict(**kwargs):
    """Combines multiple dictionaries with keys as prefix into single dictionary."""
    consolidated = {}
    for prefix, dictionary in kwargs.items():
        for key, val in dictionary.items():
            consolidated[prefix + '/' + key] = val

    return consolidated


def dict_to_str(dictionary, sep=','):
    """Converts a dictionary to string."""
    string = ''
    for i, (k, v) in enumerate(dictionary.items()):
        string += f'{k}={v}'
        if len(dictionary) != i + 1:
            string += f'{sep} '

    return string


def dict_to_df(to_convert, columns):
    """Converts a dictionary to dataframe."""
    df = pd.DataFrame.from_dict(to_convert, orient='index')
    df.reset_index(level=0, inplace=True)
    df.columns = columns
    return df


def flatten_dict(d, parent_key='', sep='_'):
    """Flattens nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
