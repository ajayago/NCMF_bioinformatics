#!/bin/bash
# All the model training
models=( "AspEm" "PTE" "metapath2vec-ESim" "ConvE" "DistMult" "HGT" "HIN2Vec" "R-GCN" "TransE" )
for i in "${models[@]}"
do
    /bin/bash "$i"/run_s1.sh > "$i"/run_s1.log 2>&1 & 
done

