# Instructions to Reproduce Baselines (DCMF)

In the steps below, the values of the placeholders can take one of the following values:
<dataset> can be {"Polypharmacy", "PubMed", "MIMIC"}
<sample_no> can be {1, 2, 3} for Polypharmacy and PubMed and {1, 3, 4} for MIMIC
<algorithm> can be "OTHER"

1. Create and initialise anaconda environment for DCMF.
    - `conda env create -f environment.yml`
    - `conda activate dcmf_env`

2. Prepare data matrices for specific dataset sample i, where i={1,2,3} for Polypharmacy and PubMed, i={1,3,4} for MIMIC. Launch the notebook at ./experiments/<dataset>_data_prep_DCMF.ipynb and update the sample id. Run the complete notebook.

3. Train DCMF by launching the notebook at ./doc/dcmf_<dataset>.ipynb and execute the entire notebook.

3. To evaluate emb.dat, copy the files to `../HNE/Model/OTHER/data`.
    - `cp emb_<dataset>_sample_<sample_no>.dat ../HNE/Model/OTHER/data/<dataset>/emb.dat`

4. Initialise anaconda environment for HNE.
    - `cd ../HNE`
    - `conda activate ./env`

5. Evaluate as with HNE [instructions](https://github.com/yangji9181/HNE/tree/0966fbb521652e1cba7a57b5b29bf81d17fec380/Evaluate). Replace <sample_no> below
    - `cd ../HNE/Evaluate`
    - `bash evaluate.sh "<dataset>" "OTHER" "<sample_no>"`

6. Repeat the above steps for all 3 samples

7. To average results over 3 samples, execute the command, after replacing dataset and algorithm name as needed:
    - `cd ../HNE/Evaluate`
    - `conda activate ../HNE/env/`
    - `python averaging_results.py <dataset> <algorithm>`
