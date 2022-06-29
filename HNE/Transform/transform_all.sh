#!/bin/bash
# All the transforms
models=( "AspEm" "PTE" "metapath2vec-ESim" "ConvE" "DistMult" "HGT" "HIN2Vec" "R-GCN" "TransE" )
for i in "${models[@]}"
do
    /bin/bash transform_"$i".sh
done
