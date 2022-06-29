import numpy as np
import pandas as pd
import os
import sys
sys.path.append("..")
from skfusion import fusion
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='DFMF')
    parser.add_argument('--sid', type=int, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    sample_id = args.sid

    data_folder = f'../../datasets/CMF/PubMed/sampled{sample_id}/'
    test_links = f'sampled{sample_id}_link.dat.test'
    emb_file = f'dfmf_s{sample_id}_emb_pubmed.dat'
    k = 50

    print('Reading Data...')
    X0 = np.load(os.path.join(data_folder, f'X0.npy'))
    X1 = np.load(os.path.join(data_folder, f'X1.npy'))
    X2 = np.load(os.path.join(data_folder, f'X2.npy'))
    X3 = np.load(os.path.join(data_folder, f'X3.npy'))
    X4 = np.load(os.path.join(data_folder, f'X4.npy'))
    X5 = np.load(os.path.join(data_folder, f'X5.npy'))
    X6 = np.load(os.path.join(data_folder, f'X6.npy'))
    X7 = np.load(os.path.join(data_folder, f'X7.npy'))
    X8 = np.load(os.path.join(data_folder, f'X8.npy'))
    X9 = np.load(os.path.join(data_folder, f'X9.npy'))

    id_idx_map = pd.read_csv(
        os.path.join(data_folder, f'id_idx.csv'),
        header=None,
        sep='\t',
        names=['id', 'idx', 'node_type']
    )
    test_links = pd.read_csv(
        os.path.join(data_folder, test_links),
        header=None,
        sep='\t',
        names=['id_from', 'id_to', 'link_status']
    )

    e0 = fusion.ObjectType('Type 1', k)
    e1 = fusion.ObjectType('Type 2', k)
    e2 = fusion.ObjectType('Type 3', k)
    e3 = fusion.ObjectType('Type 4', k)

    relations = [
        fusion.Relation(X0, e0, e0),
        fusion.Relation(X1, e0, e1),
        fusion.Relation(X2, e1, e1), 
        fusion.Relation(X3, e2, e0), 
        fusion.Relation(X4, e2, e1), 
        fusion.Relation(X5, e2, e2), 
        fusion.Relation(X6, e2, e3), 
        fusion.Relation(X7, e3, e0), 
        fusion.Relation(X8, e3, e1), 
        fusion.Relation(X9, e3, e3), 
    ]
    fusion_graph = fusion.FusionGraph()
    fusion_graph.add_relations_from(relations)

    print('Training...')
    fuser = fusion.Dfmf()
    fuser.fuse(fusion_graph)
    print(fuser.factor(e0).shape)

    print('Saving...')
    embs = []
    entities = [e0, e1, e2, e3] # must be in chronological order
    for i, e in enumerate(entities):
        id_idx = id_idx_map[id_idx_map['node_type'] == i]
        emb = pd.DataFrame(fuser.factor(e))
        embs.append(pd.merge(id_idx, emb, left_on='idx', right_index=True))

    embs = pd.concat(embs)
    embs = embs.sort_values(by=['id'])
    embs = embs.drop(columns=['idx', 'node_type'])

    params = ''
    write_path = emb_file
    with open(write_path, 'w') as file:
        file.write(f'{params}\n')
        for _, row in embs.iterrows():
            id = int(row['id'])
            emb = row[1:].astype(np.float32)
            emb_str = ' '.join(emb.astype(str))
            file.write(f'{id}\t{emb_str}\n')


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    runtime = end_time - start_time
    runtime_seconds = runtime.total_seconds()
    print(f"Total runtime = {runtime_seconds}")
