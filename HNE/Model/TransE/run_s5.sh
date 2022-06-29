#!/bin/bash
scriptdir="$(dirname "$0")"
cd "$scriptdir"

dataset="PubMed"
folder="data/${dataset}/"
node_file="${folder}node.dat"
link_file="${folder}link.dat"
rela_file="${folder}rela.dat"
emb_file="${folder}emb.dat"

mkdir bin
g++ src/transE.cpp -o bin/transE -pthread -O3 -march=native

threads=6
size=50
alpha=0.018
margin=1
epoch=400
nbatches=50

./bin/transE -entity ${node_file} -relation ${rela_file} -triplet ${link_file} -output ${emb_file} -size ${size} -out-binary 0 -epochs ${epoch} -nbatches ${nbatches} -alpha ${alpha} -margin ${margin} -threads ${threads}
