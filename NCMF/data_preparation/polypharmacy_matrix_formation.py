"""
This script uses the files obtained during the EDA to create numpy array representations of interactions between drugs and proteins.
For modelling drug-drug interactions:
Input files:
1. bio-decagon-combo.csv -- used to obtain drug-drug interactions, that cause side-effects
2. drug-pairs-without-sideeffects.csv -- used to obtain all drug pairs that do not cause any side-effects
Output file:
1. drug-drug.csv -- a 645 X 645 matrix with 
   matrix[i][j] = 1 if drug pair (i, j) interact, else 0
   
For modelling protein-protein interactions:
Input file:
1. protein-drug-distribution.csv -- used to obtain all drugs that interact with proteins - only the proteins that interact with >= 20 drugs are chosen for analysis
2. bio-decagon-ppi.csv -- for the proteins selected (based on number of drug interactions they have),this gives info on which proteins each one interacts with
Output file:
1. protein-protein.csv -- an 837 X 837 matrix with
   matrix[i][j] = 1 if protein pair (i, j) interact, else 0 

For modelling drug-protein interactions:
Input file:
1. bio-decagon-targets-all.csv -- used to obtain info on the drugs that interact with various proteins
Output file:
1. drug-protein.csv -- a 645 X 837 matrix with
   matrix[i][j] = 1 if drug i interacts with protein j, else 0

"""
import pandas as pd
import numpy as np

# drug-drug interactions
drug_df = pd.read_csv("../../datasets/NCMF/Polypharmacy/bio-decagon-combo.csv")

all_drugs = np.unique(np.append(drug_df["STITCH 1"].unique(), drug_df["STITCH 2"].unique()))

all_drugs_df = pd.DataFrame(all_drugs)
all_drugs_df.columns = ["Drug Name"]

all_drugs_df = pd.DataFrame(data = all_drugs_df.index, index = all_drugs_df["Drug Name"])
all_drugs_df.columns = ["Drug ID"]

all_drugs_dict = all_drugs_df.to_dict('index')

drug_drug_matrix = np.ones((all_drugs.shape[0], all_drugs.shape[0]))

no_sideeffect_df = pd.read_csv("../../datasets/NCMF/Polypharmacy/drug-pairs-without-sideeffects.csv")
no_sideeffect_df.head()

for i in np.array(no_sideeffect_df[["Drug1", "Drug2"]]):
    print(f"Updating entry {i}")
    d1 = i[0]
    d2 = i[1]
    drug_drug_matrix[all_drugs_dict[d1]["Drug ID"]][all_drugs_dict[d2]["Drug ID"]] = 0
    drug_drug_matrix[all_drugs_dict[d2]["Drug ID"]][all_drugs_dict[d1]["Drug ID"]] = 0

print(f"Number of non zero entries in numpy array created = {np.count_nonzero(drug_drug_matrix)}")
#print(drug_drug_matrix[0][17])
#print(drug_drug_matrix[0][1])
print("Done with drug-drug matrix!")

# protein- protein interactions
# choosing only first 837 proteins with maximum drug interactions, i.e. with more than 20 drug interactions
protein_drug_df = pd.read_csv("../../datasets/NCMF/Polypharmacy/protein-drug-distribution.csv")
protein_drug_df_less = protein_drug_df[protein_drug_df["STITCH"] >= 20].reset_index()

all_proteins = protein_drug_df_less[["Gene"]]
print(all_proteins["Gene"])
all_proteins_df = pd.DataFrame(data = all_proteins.index, index = all_proteins["Gene"])
all_proteins_df.columns = ["Protein ID"]
all_proteins_dict = all_proteins_df.to_dict('index')

protein_protein_matrix = np.zeros((len(all_proteins), len(all_proteins)))

ppi_df = pd.read_csv("../../datasets/NCMF/Polypharmacy/bio-decagon-ppi.csv")
for i in np.array(ppi_df[["Gene 1", "Gene 2"]]):
    print(f"updating entry {i}")
    p1 = i[0]
    p2 = i[1]
    if all_proteins_dict.get(p1) and all_proteins_dict.get(p2):
        protein_protein_matrix[all_proteins_dict[p1]["Protein ID"]][all_proteins_dict[p2]["Protein ID"]] = 1
        protein_protein_matrix[all_proteins_dict[p2]["Protein ID"]][all_proteins_dict[p1]["Protein ID"]] = 1

print(f"Number of non zero entries in protein-protein array created = {np.count_nonzero(protein_protein_matrix)}")
print("Done with protein-protein matrix!")

# drug protein interaction
drug_protein_df = pd.read_csv("../../datasets/NCMF/Polypharmacy/bio-decagon-targets-all.csv")
drug_protein_matrix = np.zeros((all_drugs.shape[0], len(all_proteins)))

for i in np.array(drug_protein_df[["STITCH", "Gene"]]):
    print(f"Updating entry {i}")
    d = i[0]
    p = i[1]
    if all_proteins_dict.get(p) and all_drugs_dict.get(d):
        drug_protein_matrix[all_drugs_dict[d]["Drug ID"]][all_proteins_dict[p]["Protein ID"]] = 1

print(f"Number of non zero entries for drug-protein matrix = {np.count_nonzero(drug_protein_matrix)}")
print("Done with drug-protein matrix!")

# Saving matrices as csv file
np.savetxt('../../datasets/NCMF/Polypharmacy/drug-drug.csv', drug_drug_matrix, delimiter = ",")
np.savetxt('../../datasets/NCMF/Polypharmacy/protein-protein.csv', protein_protein_matrix, delimiter = ",")
np.savetxt('../../datasets/NCMF/Polypharmacy/drug-protein.csv', drug_protein_matrix, delimiter = ',')
