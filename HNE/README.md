# Instructions to Reproduce Baselines

In the steps below, the values of the placeholders can take one of the following values:
<dataset> can be {"Polypharmacy", "PubMed", "MIMIC"}
<sample_no> can be {1, 2, 3} for Polypharmacy and PubMed and {1, 3, 4} for MIMIC
<algorithm> can be {"AspEm", "ConvE", "DistMult", "HGT", "HIN2Vec", "metapath2vec-ESim", "PTE", "R-GCN", "TransE", "ComplEx"}

#### Dataset

Once the data is available at the path ../datasets/HNE/, ensure the following path has the necessary data for MIMIC, Polypharmacy and PubMed, as shown below:

```
(base) user@server:~/<path to code repo>/HNE$ ll ./Data/
total 20
drwxrwxr-x 5 user user 4096 Sep  1 14:51 ./
drwxrwxr-x 6 user user 4096 Sep  1 14:51 ../
drwxrwxr-x 6 user user 4096 Sep  1 23:35 MIMIC/
drwxrwxr-x 2 user user 4096 Sep  1 23:34 Polypharmacy/
drwxrwxr-x 3 user user 4096 Sep  1 23:20 PubMed/
```


#### Execution

1. Create and initialise anaconda environment.
    - `wget https://drive.google.com/file/d/199V7b3B68AI2ZgvH4dtbVHHBP4vmK8mH/view?usp=sharing`
    - `tar -xvzf HNE_env.tar.gz`
    - `conda activate ./env`

2. Ensure the dataset is available at the directory ./Data/
 
3. Load dataset sample i, where i={1,2,3} for Polypharmacy and PubMed, i={1,3,4} for MIMIC. Ensure the dataset path is updated in load_sample.sh
    - `bash load_sample.sh <sample_no>`
    - The script renames the sample files related to sample i in Data to the required file names.

4. Transform data to algorithm input format for all baselines. Ensure each of the transform_xx.sh file refers to the correct dataset.
    - `bash transform_all.sh`

Note: For TransE, before running step 5, edit the file at Model/TransE/data/<dataset>/link.dat and update the first line to match the number of pairs in the file.

5. Train model as with HNE [instructions](https://github.com/yangji9181/HNE/tree/0966fbb521652e1cba7a57b5b29bf81d17fec380/Model/AspEm) using the `run_s{i}.sh` script for PubMed and run_s1.sh for the other datasets.
    - `bash run_s1.sh`

6. Evaluate model as with HNE [instructions](https://github.com/yangji9181/HNE/tree/0966fbb521652e1cba7a57b5b29bf81d17fec380/Evaluate).
    - `cd ./HNE/Evaluate`
    - `Ensure the dataset reference and sample number is updated in evaluate_all.sh`
    - `bash evaluate_all.sh`

7. Repeat above steps for all sample values.

8. To average results over 3 samples, execute the command, after replacing dataset and algorithm name as needed:
    - `cd ./HNE/Evaluate`
    - `python averaging_results.py <dataset> <algorithm>`
