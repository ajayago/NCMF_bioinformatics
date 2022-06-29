# Instructions to Reproduce Baselines (CMF, gCMF)

In the steps below, the values of the placeholders can take one of the following values:
<dataset> can be {"Polypharmacy", "PubMed", "MIMIC"}
<sample_no> can be {1, 2, 3} for Polypharmacy and PubMed and {1, 3, 4} for MIMIC
<algorithm> can be "OTHER"

1. Create and initialise anaconda environment for CMF/gCMF.
    - `conda env create -f environment.yml`
    - `conda activate r_env`

2. Train model for specific PubMed sample i, where i={1,2,3} for Polypharmacy and PubMed, i={1,3,4} for MIMIC. To train cmf/gcmf for sample 1 run the following respectively, from the experiments folder.
    - `cd  experiments`
    - `Rscript cmf_s1_<dataset>.R` 
    - `Rscript gcmf_s1_<dataset>.R` 

3. To evaluate emb.dat, copy the files to `../HNE/Model/OTHER/data`.
    - `cp cmf_s<sample_no>_<dataset>_emb.dat ../HNE/Model/OTHER/data/<dataset>/emb.dat`

4. Initialise anaconda environment for HNE.
    - `cd ../HNE`
    - `conda activate HNE_env`

5. Evaluate as with HNE [instructions](https://github.com/yangji9181/HNE/tree/0966fbb521652e1cba7a57b5b29bf81d17fec380/Evaluate). Replace sample_no as needed
    - `cd ../HNE/Evaluate`
    - `bash evaluate.sh "<dataset>" "OTHER" "<sample_no>"`

6. Repeat the above steps for all samples

7. To average results over 3 samples, execute the command, after replacing <dataset> and algorithm name as needed:
    - `cd ../HNE/Evaluate`
    - `conda activate ../HNE/env`
    - `python averaging_results.py <dataset> <algorithm>`
