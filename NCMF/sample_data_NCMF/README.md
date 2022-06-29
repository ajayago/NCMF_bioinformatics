### NCMF for an arbitrary collection of matrices

This folder provides steps to generate an arbitrary collection of matrices, convert them into a form that is NCMF-compatible and run NCMF on the dataset to obtain representations for various entities.

#### Data Generation

1. In the NCMF conda environment, launch the notebook sample_data_generation.ipynb.
2. Update X0 and X1 generation as needed. You can even add the generation of more matrices.
3. Execute the entire notebook. The resulting matrices are saved as csv files in the data_folder mentioned in the notebook ("sample" in the default case).
4. It also generates an entity_df.csv file that lists all entities in the matrices.

#### Data Preparation for NCMF

1. In the NCMF conda environment, launch the notebook data_preparation_from_matrices_with_sampling.ipynb. Set target_matrix_index to -1 if you do not need a test data split, else provide the matrix index.
2. Update the list of csv files, data_folder, mapping of entities to matrices and the matrix which is to be sampled for a test set generation(for use in link prediction). 
3. Execute the entire notebook. The resulting files are saved in the data_folder path mentioned in the notebook ("sample" by default).

#### NCMF representation learning

1. In the NCMF conda environment, launch the notebook ncmf_documentation_and_example_usage.ipynb.
2. Update the data_dir and the dataset name("sample" by default), and any hyper-parameters as needed. Set matrix_type as real or binary based on the nature of the matrix values. This sets the loss function used in the network training as ZINB for binary/count and ZINORM for real valued matrices.
3. Execute the entire notebook. The reconstructed matrices will be available at the path data_dir/dataset_name/sample_no ("./sample/1/" in the default case). The embedding will be available at the path data_dir/dataset_name ("sample/" in the default case).

