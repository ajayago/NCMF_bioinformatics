#!/bin/bash

datafolder='Data/MIMIC/'

sid=$1
echo "Loading sample ${sid}..."

# remove old data
echo "Removing..."
rm ${datafolder}info.dat 
rm ${datafolder}label.dat 
rm ${datafolder}label.dat.test 
rm ${datafolder}link.dat 
rm ${datafolder}link.dat.test 
rm ${datafolder}meta.dat 
rm ${datafolder}node.dat

# rename
echo "Renaming..."
cp ${datafolder}sampled${sid}_info.dat ${datafolder}info.dat
cp ${datafolder}sampled${sid}_label.dat ${datafolder}label.dat
cp ${datafolder}sampled${sid}_label.dat.test ${datafolder}label.dat.test
cp ${datafolder}sampled${sid}_link.dat ${datafolder}link.dat
cp ${datafolder}sampled${sid}_link.dat.test ${datafolder}link.dat.test
cp ${datafolder}sampled${sid}_meta.dat ${datafolder}meta.dat
cp ${datafolder}sampled${sid}_node.dat ${datafolder}node.dat

