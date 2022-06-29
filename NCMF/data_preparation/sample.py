import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def sample(data, sample_frac=0.33, test_link_frac=0.25, test_label_frac=0.2, min_labels=300):
    """"Subsets and splits the given data into train and test set."""
    _, _, test_link_df, _, _, node_info_df, link_info_df, label_info_df, node_meta_df, link_meta_df, label_meta_df = data
    subset_node_df, subset_link_df, subset_label_df = subset_data_min_labels(
        data, sample_frac, min_labels)
    sampled_node_df, sampled_link_df, sampled_test_link_df, sampled_label_df, sampled_test_label_df = train_test_split_pubmed(
        subset_node_df, subset_link_df, subset_label_df, test_link_frac, test_label_frac, test_link_df)

    sampled_data = sampled_node_df, sampled_link_df, sampled_test_link_df, sampled_label_df, sampled_test_label_df, node_info_df, link_info_df, label_info_df, node_meta_df, link_meta_df, label_meta_df
    return reset_node_ids(sampled_data)


def subset_data_min_labels(data, sample_frac=0.33, min_count=300):
    """Subsets given data while ensuring minimum number of labelled data in subset."""
    max_iter, counter = 200, 0
    while True:
        sampled_node_df, sampled_link_df, sampled_label_df = subset_data(
            data, sample_frac, seed=None)
        num_labelled = len(sampled_label_df)

        if num_labelled > min_count:
            print(f'Num labelled nodes: {num_labelled}')
            break

        counter += 1
        if counter > max_iter:
            raise RuntimeError(
                f'Max iteraction reached. Unable to ensure {min_count} labels.')
    return sampled_node_df, sampled_link_df, sampled_label_df


def subset_data(data, sample_frac=0.33, seed=None):
    """Creates a subset of the given data."""
    node_df, link_df, _, label_df, test_label_df, _, _, _, _, _, _ = data

    comb_label_df = pd.concat([label_df, test_label_df])
    labelled_node_ids = comb_label_df[['node_id']].values.ravel()

    labelled_node_df = node_df[node_df['node_id'].isin(labelled_node_ids)]
    unlabelled_node_df = node_df[~node_df['node_id'].isin(labelled_node_ids)]

    num_labelled_nodes = labelled_node_ids.shape[0]
    node_counts_df = pd.DataFrame(data={
        'node_type': np.arange(len(node_df['node_type'].value_counts())),
        'original_count': node_df['node_type'].value_counts().sort_index()
    })
    node_counts_df['sample_count'] = node_counts_df['original_count'] * sample_frac
    node_counts_df['sample_count'] = node_counts_df['sample_count'].astype(int)
    node_counts_df['sample_count'][node_counts_df['node_type'] ==
                                   1] = node_counts_df['sample_count'][node_counts_df['node_type'] == 1] - num_labelled_nodes

    tmp_node_dfs = [labelled_node_df]  # include all labeled nodes
    for node_type, _, n in node_counts_df.itertuples(index=False):
        if node_type == 0:
            filtered_links = link_df[(link_df['link_type'] == 0) | (
                link_df['link_type'] == 1)]  # GENE-GENE or GENE-DISEASE
            gene_node_ids = sample_high_out_deg(
                filtered_links, topk_percentage=0.5, sample_count=n)
            tmp_df = unlabelled_node_df[unlabelled_node_df['node_id'].isin(
                gene_node_ids)]
        else:
            tmp_df = unlabelled_node_df[unlabelled_node_df['node_type'] == node_type].sample(
                n, random_state=seed)
        tmp_node_dfs.append(tmp_df)
    sampled_node_df = pd.concat(tmp_node_dfs)

    sampled_link_df = link_df[link_df['node_id_from'].isin(
        sampled_node_df['node_id']) & link_df['node_id_to'].isin(sampled_node_df['node_id'])]

    # keep nodes with at least 1 incoming / outgoing link
    node_ids_with_links = pd.unique(
        sampled_link_df[['node_id_from', 'node_id_to']].values.ravel())
    sampled_node_df = sampled_node_df[sampled_node_df['node_id'].isin(
        node_ids_with_links)]
    sampled_label_df = comb_label_df[comb_label_df['node_id'].isin(
        sampled_node_df['node_id'])]
    return sampled_node_df, sampled_link_df, sampled_label_df


def sample_high_out_deg(link_df, topk_percentage, sample_count):
    """Samples nodes with preference for high out degree."""
    link_outdegree = link_df.groupby(['node_id_from'])[
        'node_id_from'].count().reset_index(name='count')
    topk = int(len(link_outdegree) * topk_percentage)
    high_out_node_ids = link_outdegree.nlargest(
        topk, 'count')['node_id_from'].values
    # TODO: may need to select topk and sample the remaining with equal outdegree
    sampled_node_ids = np.random.choice(high_out_node_ids, sample_count)
    return sampled_node_ids


def train_test_split_pubmed(node_df, link_df, label_df, test_link_frac, test_label_frac, test_link_df, seed=None):
    """"Splits the given data in the pubmed data format into train and test sets"""
    sampled_node_df = node_df
    sampled_link_df, sampled_test_link_df = train_test_split_link(
        node_df, link_df, test_link_frac)
    sampled_label_df, sampled_test_label_df = train_test_split(
        label_df, test_size=test_label_frac, random_state=seed, stratify=label_df[['node_label']])
    return sampled_node_df, sampled_link_df, sampled_test_link_df, sampled_label_df, sampled_test_label_df


def train_test_split_link(node_df, link_df, test_link_frac=0.25):
    """Splits the links (interactions) into train and test sets."""
    all_node_occs = link_df[link_df['link_type'] == 1][[
        'node_id_from', 'node_id_to']].values.ravel()
    unique, counts = np.unique(all_node_occs, return_counts=True)
    all_node_occs_count = dict(zip(unique, counts))

    num_test_links = int(
        len(link_df[link_df['link_type'] == 1]) * test_link_frac)
    count = 0
    train_link_df = link_df.copy()
    pos_test_link_dfs = []
    while count < num_test_links:
        pos_test_link = train_link_df[train_link_df['link_type'] == 1].sample(
            1)
        node_id_from = pos_test_link['node_id_from'].values[0]
        node_id_to = pos_test_link['node_id_to'].values[0]

        if all_node_occs_count[node_id_from] < 2 or all_node_occs_count[node_id_to] < 2:
            continue

        pos_test_link_dfs.append(pos_test_link)
        train_link_df = train_link_df.drop(index=pos_test_link.index)

        all_node_occs_count[node_id_from] -= 1
        all_node_occs_count[node_id_to] -= 1
        count += 1

    pos_test_link_df = pd.concat(pos_test_link_dfs)
    pos_test_link_df = pos_test_link_df[[
        'node_id_from', 'node_id_to', 'link_weight']]
    pos_test_link_df = pos_test_link_df.rename(
        columns={'link_weight': 'link_status'})
    pos_test_link_df['link_status'] = 1

    disease_node_ids = node_df[node_df['node_type'] == 1]['node_id'].values
    neg_test_link_dfs = []
    gd_deg = pos_test_link_df.groupby('node_id_from')[
        'node_id_from'].count().reset_index(name='out_degree')
    for node_from, n_links in gd_deg.itertuples(index=False):
        filtered_pos_links = link_df[(link_df['link_type'] == 1) & (
            link_df['node_id_from'] == node_from)]
        filtered_neg_links = disease_node_ids[~np.isin(
            disease_node_ids, filtered_pos_links['node_id_to'])]

        sampled_neg_links = np.random.choice(filtered_neg_links, n_links)
        sampled_neg_links = pd.DataFrame({
            'node_id_from': [node_from for _ in range(n_links)],
            'node_id_to': sampled_neg_links.tolist(),
            'link_status': [0 for _ in range(n_links)],
        })
        neg_test_link_dfs.append(sampled_neg_links)

    neg_test_link_df = pd.concat(neg_test_link_dfs)

    test_link_df = pd.concat([pos_test_link_df, neg_test_link_df])
    test_link_df = test_link_df.reset_index(drop=True)
    return train_link_df, test_link_df


def reset_node_ids(data):
    """Resets the node ids of the data such that they are in chronological order from 1 to N."""
    node_df, link_df, test_link_df, label_df, test_label_df, node_info_df, link_info_df, label_info_df, node_meta_df, link_meta_df, label_meta_df = data

    node_df = node_df.sort_values('node_type')
    link_df = link_df.sort_values('link_type')

    old_node_ids = node_df['node_id'].to_list()
    new_node_ids = [i for i in range(len(node_df))]
    old_new_ids = dict(zip(old_node_ids, new_node_ids))

    node_df['node_id'] = node_df['node_id'].apply(lambda x: old_new_ids[x])
    node_df = node_df.reset_index(drop=True)

    link_df['node_id_from'] = link_df['node_id_from'].apply(
        lambda x: old_new_ids[x])
    link_df['node_id_to'] = link_df['node_id_to'].apply(
        lambda x: old_new_ids[x])
    link_df = link_df.reset_index(drop=True)
    test_link_df['node_id_from'] = test_link_df['node_id_from'].apply(
        lambda x: old_new_ids[x])
    test_link_df['node_id_to'] = test_link_df['node_id_to'].apply(
        lambda x: old_new_ids[x])
    test_link_df = test_link_df.reset_index(drop=True)

    label_df['node_id'] = label_df['node_id'].apply(lambda x: old_new_ids[x])
    label_df = label_df.reset_index(drop=True)
    test_label_df['node_id'] = test_label_df['node_id'].apply(
        lambda x: old_new_ids[x])
    test_label_df = test_label_df.reset_index(drop=True)
    return node_df, link_df, test_link_df, label_df, test_label_df, node_info_df, link_info_df, label_info_df, node_meta_df, link_meta_df, label_meta_df
