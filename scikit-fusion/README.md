# Instructions to Reproduce Baselines (dfmf)

In the steps below, the values of the placeholders can take one of the following values:
<dataset> can be {"Polypharmacy", "PubMed", "MIMIC"}
<sample_no> can be {1, 2, 3} for Polypharmacy and PubMed and {1, 3, 4} for MIMIC
<algorithm> can be "OTHER"

1. Create and initialise anaconda environment for dfmf.
    - `wget https://drive.google.com/file/d/1yscDEaXFPpL2gc8z_gSP7sJUDX13fkyU/view?usp=sharing`
    - `tar -xvzf DFMF_env.tar.gz`
    - `conda activate ./env`

2. rain model for specific dataset sample i, where i={1,2,3} for Polypharmacy and PubMed, i={1,3,4} for MIMIC. To train dfmf for sample 1 run the following respectively. 
    - `python dfmf_<dataset>.py --sid 1`

3. To evaluate emb.dat, copy the files to `../HNE/Model/OTHER/data`.
    - `cp dfmf_s<sample_no>_emb_<dataset>.dat ../HNE/Model/OTHER/data/<dataset>/emb.dat`

4. Initialise anaconda environment for HNE.
    - `cd ../HNE`
    - `conda activate ./env`

5. Evaluate as with HNE [instructions](https://github.com/yangji9181/HNE/tree/0966fbb521652e1cba7a57b5b29bf81d17fec380/Evaluate). Replace sample_no as needed
    - `cd ../HNE/Evaluate`
    - `bash evaluate.sh "<dataset>" "OTHER" "<sample_no>"`

6. To average results over 3 samples, execute the command, after replacing dataset and algorithm name as needed:
    - `cd ../HNE/Evaluate`
    - `conda activate ../HNE/env/`
    - `python averaging_results.py <dataset> <algorithm>`
