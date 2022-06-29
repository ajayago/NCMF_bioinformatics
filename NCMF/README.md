NCMF:
=====

This folder contains all relevant folders to obtain NCMF embeddings for various datasets, like PubMed, Decagon Polypharmacy, MIMIC etc, followed by a downstream link prediction task to evaluate the embeddings generated.

NCMF input and output:
=====================
All input files must be present in the folder ../datasets/NCMF/<dataset>/. The following files are necessary for NCMF to work:
1. sampled<sample_no>_node.dat
2. sampled<sample_no>_link.dat
3. sampled<sample_no>_link.dat.test
4. sampled<sample_no>_label.dat
5. sampled<sample_no>_label.dat.test
6. sampled<sample_no>_meta.dat
7. sampled<sample_no>_info.dat

The output from NCMF is the embedding file called emb_<dataset>_<sample_no>.dat at the location ../datasets/NCMF/<dataset>/

The results of the link prediction evaluation task is recorded in the file ../datasets/NCMF/<dataset>/sampled<sample_no>_record.dat

Folder structure and hierarchy:
===============================

./

|

 -- doc/ 

|

-- src/

|

-- experiments/

doc folder - holds scripts to run NCMF for arbitrary datasets

src folder - contains all relevant functions and classes to perform NCMF training and evaluation

experiments folder - contains the files to run representation learning for various datasets

Representation Learning:
========================

In the steps below, the values of the placeholders can take one of the following values:
<dataset> can be {"Polypharmacy", "PubMed", "MIMIC"}
<sample_no> can be {1, 2, 3} for Polypharmacy and PubMed and {1, 3, 4} for MIMIC
<algorithm> can be "NCMF"

Set up the environment using conda by running the command:

```
conda env create -f environment.yml
conda activate ncmf
```

1. Place the input data files at the path ../datasets/NCMF/<dataset>/

2. Launch the file ncmf_documentation_and_example_usage.ipynb using Jupyter notebook and update the sample number, dataset name and path to the dataset directory as needed. Update the hyperparameters if needed. Execute the entire notebook.

3. The reconstructed matrices are saved at the path ../datasets/NCMF/<dataset>/<sample_no>. Ensure this path exists prior to running the notebook.

4. Evaluate the AUC and MRR scores of the embeddings learnt by running a link prediction task, using the notebook ncmf_evaluation_example_usage.ipynb

Hyperparameter Tuning with Ax:
==============================

Set up the environment using conda by running the command:

```
conda env create -f environment_ax.yml
conda activate ax
```

1. Place the input data files at the path ../datasets/NCMF/<dataset>/

2. Launch the file ncmf_documentation_and_example_usage_with_ax.ipynb using Jupyter notebook and update the sample number, dataset name and path to the dataset directory as needed. Update the hyperparameters to be tuned and the range to be explored. Execute the entire notebook.

3. The reconstructed matrices are saved at the path ../datasets/NCMF/<dataset>/<sample_no>. Ensure this path exists prior to running the notebook.

Experiments:
===========

The folder ./experiments has the following files to reproduce the NCMF representation learning for various datasets:

#### MIMIC dataset

1. Ensure the data files are available at the path ../datasets/NCMF/MIMIC/. Ensure the folders 1, 3 and 4 exist in this path - the embeddings will be saved here.

2. From the ./experiments folder, launch the notebook ncmf_mimic.ipynb and update the sample number (1, 3, 4 are the allowed values), and the hyperparameters(if needed). Execute the entire notebook.

3. Check the embeddings by running a link prediction task, using the notebook ncmf_evaluation_example_usage.ipynb, after updating the sample number and dataset name.

4. Repeat the above three steps for all samples.

5. To average over all 3 samples, navigate to src/ and run the command
`python averaging_results.py MIMIC NCMF`

#### Polypharmacy dataset - PolyP1

1. Ensure the data files are available at the path ../datasets/NCMF/ESP/. Ensure the folder 1 exists in this path - embeddings will be saved here.

2. To run DCA on the drug x drug side-effect matrix, follow the README at data_preparation/dca path, after preparing the original data using the notebook data_preparation_NCMF_esp_pseudo_polypharmacy.ipynb.

3. After DCA execution, to run NCMF on the drug x latent matrix, drug x protein and protein x protein matrices, use data_preparation/data_prep_ESP.ipynb to get the files needed for NCMF.

4. Execute the notebook experiments/ncmf_esp.ipynb to get drug and protein representations.

5. To evaluate the representations learnt, run the notebook ESP_evaluation.ipynb

#### Polypharmacy dataset - PolyP2

1. Ensure the data files are available at the path ../datasets/NCMF/Polypharmacy/. Ensure the folders 1, 2 and 3 exist in this path - the embeddings will be saved here.

2. From the ./experiments folder, launch the notebook ncmf_polypharmacy.ipynb and update the sample number (1, 2, 3 are the allowed values),  and the hyperparameters(if needed). Execute the entire notebook.

3. Check the embeddings by running a link prediction task, using the notebook ncmf_evaluation_example_usage.ipynb, after updating the sample number and datas
et name.

4. Repeat the above three steps for all samples.

5. To average over all 3 samples, navigate to src/ and run the command
`python averaging_results.py Polypharmacy NCMF`

#### PubMed dataset

1. Ensure the data files are available at the path ../datasets/NCMF/PubMed/. Ensure the folders 1, 2 and 3 exist in this path - the embeddings will be saved here.

2. From the ./experiments folder, launch the notebook ncmf_pubmed.ipynb and update the sample number (1, 2, 3 are the allowed values),  and the hyperparameters(if needed). Execute the entire notebook.

3. Check the embeddings by running a link prediction task, using the notebook ncmf_evaluation_example_usage.ipynb, after updating the sample number and datas
et name.

4. Repeat the above three steps for all samples.

5. To average over all 3 samples, navigate to src/ and run the command
`python averaging_results.py PubMed NCMF`
