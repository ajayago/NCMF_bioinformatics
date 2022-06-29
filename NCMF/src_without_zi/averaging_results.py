import pandas as pd
import numpy as np
import sys

dataset = sys.argv[1]
sample_mapping = {"MIMIC": [1, 3, 4],
                  "Polypharmacy": [1, 2, 3],
                  "PubMed": [1, 2, 3]}
algorithm = sys.argv[2]

total_auc = []
total_mrr = []

for s in sample_mapping[dataset]:
    filename = f"../../datasets/NCMF/out/{dataset}_sample{s}_{algorithm}_results.csv"
    df = pd.read_csv(filename)
    total_auc.append(df['AUC'][0])
    total_mrr.append(df['MRR'][0])

avg_auc = sum(total_auc)/3
avg_mrr = sum(total_mrr)/3

std_auc = np.std(total_auc)
std_mrr = np.std(total_mrr)

print(f"Statistics for {dataset} - {algorithm}")
print("=======================================")
print(f"AUC = {avg_auc:.4f} +/- {std_auc:.4f}")
print(f"MRR = {avg_mrr:.4f} +/- {std_mrr:.4f}")

with open(f"../../datasets/NCMF/out/{dataset}_avg_{algorithm}_results.txt", 'w') as outfile:
    outfile.write(f"Statistics for {dataset} - {algorithm}")
    outfile.write("\n")
    outfile.write("=======================================\n")
    outfile.write(f"AUC = {avg_auc:.4f} +/- {std_auc:.4f}")
    outfile.write("\n")
    outfile.write(f"MRR = {avg_mrr:.4f} +/- {std_mrr:.4f}")
    outfile.write("\n")

