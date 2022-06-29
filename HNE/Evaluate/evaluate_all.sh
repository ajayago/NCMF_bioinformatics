#!/bin/bash
# All the model training
# models=( "HIN2Vec" "metapath2vec-ESim" "R-GCN" "ConvE" "TransE" )
models=( "HGT" )
for i in "${models[@]}"
do
    /bin/bash ./evaluate.sh "CellLine" "$i" "3"
done
