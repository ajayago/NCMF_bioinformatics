DCA:
===
This folder contains code sourced from https://github.com/theislab/dca, which is used in our NCMF experiment to learn representations for the drug x drug side effect matrix.

Execute the following steps to run DCA:
```
Download the file from https://drive.google.com/file/d/1GMvhMl_z9rmy9hzjtTPPXp66AK6puW29/view?usp=sharing and place it in the current directory
bzip2 -d DCA_input_data.tar.bz2
tar -xvf DCA_input_data.tar
conda env create -f environment.yml
pip install dca
dca X0_full_dca.csv results_reduced_drug_se
dca X0_full_dca_transpose.csv results_reduced_drug
```
