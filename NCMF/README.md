NCMF:
=====

This folder contains all relevant folders to obtain NCMF embeddings for various datasets, like PubMed, Decagon Polypharmacy, MIMIC etc, followed by a downstream link prediction task to evaluate the embeddings generated.

NCMF input and output:
=====================
All input files must be present in the folder ./datasets/<dataset_name>/. The following files are necessary for NCMF to work:
1. sampled<sample_number>_node.dat
2. sampled<sample_number>_link.dat
3. sampled<sample_number>_link.dat.test
4. sampled<sample_number>_label.dat
5. sampled<sample_number>_label.dat.test
6. sampled<sample_number>_meta.dat
7. sampled<sample_number>_info.dat

The output from NCMF is the embedding file called emb_<dataset_name>_<sample_number>.dat at the location ./datasets/<dataset_name>/

The results of the link prediction evaluation task is recorded in the file ./datasets/<dataset_name>/sampled<sample_number>_record.dat

Folder structure and hierarchy:
===============================

./

|

 -- data_preparation/ 

|

 -- datasets/

|

-- src/

|

-- experiments/

data_preparation folder - holds scripts to convert data in the form of triplets and matrices into a format that can be used by NCMF

datasets folder - holds all samples generated from the datasets on which experiments were carried out 

src folder - contains all relevant functions and classes to perform NCMF training and evaluation

experiments folder - contains the files to run representation learning for various datasets

Representation Learning:
========================

Set up the environment using conda by running the command:

```
conda env create -f environment.yml
```

1. Place the input data files at the path ./datasets/<dataset_name>/

2. Launch the file ncmf_documentation_and_example_usage.ipynb using Jupyter notebook and update the sample number, dataset name and path to the dataset directory as needed. Update the hyperparameters if needed. Execute the entire notebook.

3. The reconstructed matrices are saved at the path ./datasets/<dataset_name>/<sample_number>. Ensure this path exists prior to running the notebook.

4. Evaluate the AUC and MRR scores of the embeddings learnt by running a link prediction task, using the notebook ncmf_evaluation_example_usage.ipynb

Experiments:
===========

Copy the files to the outer directory before executing them.

ncmf_documentation_and_example_usage_with_ax.ipynb -- This notebook can be used to combine Ax for finding the best set of hyperparameters.

The folder ./experiments has the following files to reproduce the NCMF representation learning for various datasets:

ncmf_simulated.ipynb -- Representation learning for the simulated dataset experiment

ncmf_mimic.ipynb -- Representation learning for the MIMIC dataset

ncmf_polypharmacy.ipynb -- Representation learning for the Polypharmacy dataset

ncmf_pubmed.ipynb -- Representation learning for the PubMed dataset

ncmf_evaluation_simulated.ipynb -- For the evaluation of the embeddings learnt from the simulated dataset experiment

rmse_visualization_simdata.ipynb -- For visualization of the RMSE values for the simulated dataset

Simulation study:
================

- We generated the following 3 datasets 

`ncmf_sim_study/ncmf_sim_data/dict_name_dataset_1.pkl`
`ncmf_sim_study/ncmf_sim_data/dict_name_dataset_2.pkl`
`ncmf_sim_study/ncmf_sim_data/dict_name_dataset_3.pkl`
using the scripts below (by changing the random seed to 0, 10 and 20 respectively in ncmf_sim_data_generator.py)

`ncmf_sim_study/"1 - syn_data_gen - generate required datasets - try 1 - final.ipynb"`
`ncmf_sim_study/ncmf_sim_data_generator.py`

- The results from the paper can be obtained by running the following scripts for each dataset (by changing the dataset filepath in "2 - data_preparation_from_matrices...ipynb") in the same order to obtain the results for NCMF and the baselines: CMF, gCMF, DCMF and DFMF.
	- ncmf_sim_study/"2 - data_preparation_from_matrices_with_sampling- aug multiview setup - test - try1 - final- ALL datasets - try1.ipynb"
	- ncmf_sim_study/"3 - ncmf_documentation_and_example_usage - aug multiview - try1.ipynb"
	- ncmf_sim_study/"3 - ncmf_documentation_and_example_usage - aug multiview - try1.py"
	- ncmf_sim_study/"4 -  ncmf_sim_results_gen - try 2 final.ipynb"
	- ncmf_sim_study/"5 - data_prep_cmf - 4 - final.ipynb"
	- ncmf_sim_study/6_cmf_1.R
	- ncmf_sim_study/6_gcmf_1.R
	- ncmf_sim_study/"7 -  cmf_sim_results_gen - try2 - final.ipynb"
	- ncmf_sim_study/"7 -  gcmf_sim_results_gen - try2 - final.ipynb"
	- ncmf_sim_study/dcmf/doc/"8 - ncmf_dcmf_expt - try 2 - final.ipynb"
	- ncmf_sim_study/dcmf/doc/"9 - dcmf_sim_results_gen - try1 - final.ipynb"
	- ncmf_sim_study/dcmf/doc/"10 - ncmf_dfmf_expt - try 2 - final.ipynb"
	- ncmf_sim_study/dcmf/doc/"11 - dfmf_sim_results_gen - try1 - final.ipynb"
	- ncmf_sim_study/dcmf/doc/"12 - aggregate results - 1.ipynb"

- After obtaining the results for the 3 datasets, to obtain the aggregated results shown in the paper:
	- Copy "ncmf-main/NCMF/ncmf_sim_study/ncmf_sim_data/run_\#" to "ncmf_sim_study/sim_results_charting/run_\#", where \# is the dataset number 1, 2 or 3
	- Run "ncmf_sim_study/sim_results_charting/chart - agg 2 - try 2 - final.ipynb"

- Note:
	- To run CMF and gCMF, `R` must be installed along with the packages `CMF` and `itertools`
	- To run DFMF, `python 2.7` must be used
