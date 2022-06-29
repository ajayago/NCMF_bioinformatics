This folder has scripts to convert matrices into the format used by NCMF.

Input files:
===========

The inputs must be in the form of matrices stored as separate csv files at ../datasets/<dataset_name> path

Output files:
=============

Here, sample_number denotes the sample ID for each of the datasets. These are created at the current directory.
1. sampled<sample_number>_node.dat
2. sampled<sample_number>_link.dat
3. sampled<sample_number>_link.dat.test
4. sampled<sample_number>_label.dat
5. sampled<sample_number>_label.dat.test
6. sampled<sample_number>_meta.dat
7. sampled<sample_number>_info.dat

Helper notebooks:
=================

Some additional data pre-processing can be done using the notebooks available at this location.

data_preparation_from_matrices.ipynb -- If the data is available as matrices and need to be converted into the form of the dat files used by NCMF, use this notebook. The train and test split of the dataset must already have been done for this.

data_preparation_from_matrices_with_sampling.ipynb -- If the data is available as matrices and needs to be converted to dat file format, and if sampling needs to be done to get a train test split on the target matrix, use this notebook.

MIMIC csv creation.ipynb -- This notebook was used to create csv files for the matrices used in the MIMIC experiment.

polypharmacy_matrix_formation.py -- This notebook was used to create csv files for the matrices used in Polypharmacy experiment.

#### Polypharmacy dataset:
1. Place the dataset having the files bio-decagon-combo.csv, drug-pairs-without-sideeffects.csv, protein-drug-distribution.csv, bio-decagon-ppi.csv and bio-decagon-targets-all.csv at the path ../../datasets/NCMF/Polypharmacy.

2. Activate the environment and run the following command to generate the matrices drug-drug.csv, drug-protein.csv and protein-protein.csv
`conda activate ncmf`
`python polypharmacy_matrix_formation.py`

3. Launch the Jupyter notebook data_preparation_from_matrices_with_sampling.ipynb and update the sample_id, path to the dataset, matrix names, their mapping with entities, entity names, target matrix name. Execute the entire notebook. 

4. This creates the samples as needed by NCMF (no scaling is done)

#### MIMIC dataset:
1. Place the dataset (i.e pickle files dict_nsides_mimic_data_v1_case1_part.pkl, dict_nsides_mimic_data_v3_case1_part.pkl and dict_nsides_mimic_data_v4_case1_part.pkl) at the path ../../datasets/NCMF/MIMIC.

2. Activate the environment and execute the notebook 'MIMIC csv creation.ipynb' after updating the path and file name.

3. Launch the Jupyter notebook data_preparation_from_matrices_with_sampling.ipynb and update the sample_id, path to the dataset, matrix names, their mapping with entities, entity names, target matrix name. Execute the entire notebook.

4. This creates the samples as needed by NCMF (no scaling is done)

5. To scale the MIMIC dataset, execute the notebook MIMIC_scaling.ipynb after updating the sample number

NOTE: To obtain the processed dataset files, *dict_nsides_mimic_data_v<sample_number>_case1_part.pkl*, from the original MIMIC and NSIDES databases, use *NSIDES_MIMIC_data_gen.py*. Download the dependent data files "pubmed_datagen_dependent_data_files.tar.gz" and untar in the data_preparation directory before executing *NSIDES_MIMIC_data_gen.py*. Download link: https://drive.google.com/file/d/1iWl_ltXN88AVembQ_pIcWpfuS9K2Cd1g/view?usp=sharing

#### PubMed dataset:
1. Please place the complete orignal PubMed dataset at the path ../../datasets/NCMF/PubMed.

2. Launch the Jupyter notebook PubMed_sample_data_generation.ipynb and update the sample_id, path to the dataset and execute the entire notebook.

