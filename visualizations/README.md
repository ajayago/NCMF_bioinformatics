This folder contains notebooks for obtaining the following, after the embeddings are learnt using various techniques:

1. visualizations_<dataset>.ipynb -- Run this notebook to visualize the clusters for various entities obtained using hypertools for the dataset.
2. AUC_MRR_comparisons_HNE.ipynb -- Run this notebook to get baar plots comparing AUC and MRR for the HNE methods
3. AUC_MRR_comparisons_MF.ipynb -- Run this notebook to get baar plots comparing AUC and MRR for the matrix factorization methods

To execute the notebooks:
```
cd ../NCMF
conda env create -f environment.yml
conda activate ncmf
cd ../visualizations/
Download the file from the path https://drive.google.com/file/d/1XZ8gxO3Ufgg12YI597DnhSFu8KshoAg5/view?usp=sharing and place in the current directory.
bzip2 -d visualization_files.tar.bz2
tar -xvf visualization_files.tar
```
Execute the notebooks to obtain the visualizations.
