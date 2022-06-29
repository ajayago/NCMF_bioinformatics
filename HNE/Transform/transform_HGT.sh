#!/bin/bash
scriptdir="$(dirname "$0")"
cd "$scriptdir"


# Note: Only 'R-GCN', 'HAN', and 'HGT' support attributed='True' or supervised='True'
# Note: Only 'DBLP' and 'PubMed' contain node attributes.

dataset='MIMIC' # choose from 'DBLP', 'Yelp', 'Freebase', and 'PubMed'
model='HGT' # choose from 'metapath2vec-ESim', 'PTE', 'HIN2Vec', 'AspEm', 'HEER', 'R-GCN', 'HAN', 'HGT', 'TransE', 'DistMult', and 'ConvE'
attributed='False' # choose 'True' or 'False'
supervised='False' # choose 'True' or 'False'

mkdir ../Model/${model}/data
mkdir ../Model/${model}/data/${dataset}

python transform.py -dataset ${dataset} -model ${model} -attributed ${attributed} -supervised ${supervised}
