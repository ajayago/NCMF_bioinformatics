
import pandas as pd
import numpy as np
import random
import pickle as pkl
import sys
import itertools
from icd9_master import icd9
import time
from scipy.sparse import coo_matrix
from collections import Counter
import time


dname = "/data/ragu/MIMIC3/"
dname_nsides = "/data/ragu/heuristic/ade_data_nsides/"
#
out_dname = "/data/ragu/heuristic/ade_data_nsides/" 
version_name = "5"

fname_diagnosis = dname + "DIAGNOSES_ICD.csv"
fname_prescrip = dname + "PRESCRIPTIONS.csv"
fname_admissions = dname + "ADMISSIONS.csv"
#
fname_drugs = dname_nsides + "nsides_rxcui_icd9.csv"
fname_map = "/data/ragu/heuristic/ade_data_/ATC_NDC_map/FDA NDC directory with atc5 atc4 ingredients (2020_06_17)/ndc_map 2020_06_17 (atc5 atc4 ingredients).csv" 
fname_drugs_rxnorm_rxcui_ndc = "/data/ragu/umls/subsets/RXNSAT_NDC.RRF"

#mimic
df_diag = pd.read_csv(fname_diagnosis)
df_pres = pd.read_csv(fname_prescrip,dtype={"NDC":str})
df_drugs = pd.read_csv(fname_drugs,dtype={"RXCUI":str,"ICD9_CODE":str})
df_admn = pd.read_csv(fname_admissions)
#ext work
df_map = pd.read_csv(fname_map,dtype={"in_rxcui":str})
#umls
df_drugs_rxnorm_rxcui_ndc = pd.read_csv(fname_drugs_rxnorm_rxcui_ndc,sep="|",header=None)
df_drugs_rxnorm_rxcui_ndc.columns = ["RXCUI","LUI","SUI","RXAUI","STYPE","CODE","ATUI","SATUI","ATN","SAB","ATV","SUPPRESS","CVF","NA"]


#############

#from: https://github.com/chb/py-umls/blob/d4ba79245b146c1a59363668d0ef4918689c25d1/rxnorm.py#L45
def ndc_normalize(ndc):
    """ Normalizes an NDC (National Drug Code) number.

    The pseudo-code published by NIH
    (http://www.nlm.nih.gov/research/umls/rxnorm/NDC_Normalization_Code.rtf)
    first identifies the format (e.g. "6-3-2") and then normalizes based on
    that finding. However since the normalized string is always 5-4-2,
    padded with leading zeroes and removing all dashes afterwards, this
    implementation goes a much simpler route.

    NDCs that only contain one dash are treated as if they were missing the
    package specifier, so they get a "-00" appended before normalization.

    :param str ndc: The NDC to normalize as string
    :returns: A string with the normalized NDC, or `None` if the number
        couldn't be normalized
    """
    if ndc is None or 0 == len(ndc) or len(ndc) > 14:
        return None

    # replace '*' with '0' as some of the NDCs from MTHFDA contain * instead of 0
    norm = ndc.replace('*', '0')

    # split at dashes, pad with leading zeroes, cut to desired length
    parts = norm.split('-')

    # Code with only one dash; this is NOT mentioned in the above cited
    # reference but I see a lot of codes with 6-4 format.
    # These are likely codes without package specifier, though some that I
    # checked seem to not or no longer exist.
    # We append "-00" to get a 6-4-2 format and are done with it.
    if 2 == len(parts):
        parts.append('00')

    # two dashes, 6-4-1 or 5-3-2 or similar formats, concat to 5-4-2
    if 3 == len(parts):
        norm = '{}{}{}'.format(('00000'+parts[0])[-5:], ('0000'+parts[1])[-4:], ('00'+parts[2])[-2:])

    # no dashes
    elif 1 == len(parts):

        # "if NDC passed has 12 digits and first char is '0' and it's from
        # VANDF then trim first char". We do NOT check if it's from the VA
        # as this would require more information than just the NDC
        if 12 == len(norm) and '0' == norm[:1]:
            norm = norm[1:]

        # only valid if it's 11 digits
        elif 11 != len(norm):
            return None

    # reject NDCs that still contain non-numeric chars
    return norm if norm.isdigit() else None

##################

#Add ndc_norm to external data "df_map" and subset only the reqd columns

list_ndc_norm = []
for idx,row in df_map.iterrows():
    cur_ndc = row["ndc"]
    cur_ndc_norm = ndc_normalize(cur_ndc)
    list_ndc_norm.append(cur_ndc_norm)

df_map["NDC_NORM"] = list_ndc_norm

df_map_cols_subset = df_map[["in_rxcui","NDC_NORM"]].drop_duplicates()


#Add ndc_norm to MIMIC prescriptions "df_pres" and subset only the reqd columns

df_pres = pd.merge(left=df_pres,right=df_map_cols_subset,\
                          how='left',left_on="NDC",right_on="NDC_NORM").drop_duplicates()

print("#unique drugs: ")
print("#in map: ")
print("df_map_cols_subset[\"in_rxcui\"].unique().shape: ",df_map_cols_subset["in_rxcui"].unique().shape)
print("#in mimic after mapping: ")
print("df_pres[\"in_rxcui\"].unique().shape: ",df_pres["in_rxcui"].unique().shape)
print("#")


print("df_diag.columns: ")
print(df_diag.columns)
print("#")

print("df_pres.columns: ")
print(df_pres.columns)
print("#")

print("df_drugs.columns: ")
print(df_drugs.columns)
print("#")

print("df_admn.columns: ")
print(df_admn.columns)
print("###")

print("df_diag.dtypes: ")
print(df_diag.dtypes)
print("#")

print("df_pres.dtypes: ")
print(df_pres.dtypes)
print("#")

print("df_drugs.dtypes: ")
print(df_drugs.dtypes)
print("#")

print("df_admn.dtypes: ")
print(df_admn.dtypes)
print("###")


#Data gen - start

#entities involved vs data involved
# patients p         - df_diag, df_pres
# drugs r            - df_drugs, df_pres
# dis side-effects s - df_drugs
# dis treated t      - df_diag

#codes to use
# drugs - in_rxcui/RXCUI
# disease - ICD9_CODE


# 1 - find common drugs - NSIDES & MIMIC
print("unique #drugs in df_drugs: ")
print(df_drugs["RXCUI"].unique().shape) #1322
print("#")
print("unique #drugs in df_pres: ")
print(df_pres["in_rxcui"].unique().shape) #4205
print("#")
print("#common drugs: ")
drugs_common = list(set(df_drugs["RXCUI"].unique().tolist()).intersection(set(df_pres["in_rxcui"].unique().tolist())))
drugs_common = list(drugs_common)
print("len(np.unique(drugs_common)): ",len(np.unique(drugs_common))) 
print("#")
print("###")


# 2 - find common drugs i.e. betwn NISDES side effects \
# & MIMIC diagnosis
print("#diseases: df_drugs ")
print(df_drugs["ICD9_CODE"].unique().shape) #1298
print("#")
print("#diseases: df_diag")
print(df_diag["ICD9_CODE"].unique().shape) #6985
print("#")
print("#common diseases: ")
dis_common = list(np.unique(list(set(df_drugs["ICD9_CODE"].unique().tolist()).intersection(set(df_diag["ICD9_CODE"].unique().tolist())))))
print("len(np.unique(dis_common)): ",len(np.unique(dis_common)))
print("#")
print("###")


# 3 - sample patients 

# 3a - find patients who were administered atleast one drug
# from drugs_common - using prescriptions data df_pres

df_pres_drugs_common = df_pres[df_pres["in_rxcui"].isin(drugs_common)]

df_pres_pat_count = df_pres_drugs_common.groupby(["SUBJECT_ID"])["in_rxcui"]\
								 .agg(['count']) \
								 .sort_values(['count'], ascending=False)

#keep only patients who were administered (i) drugs with atleast 5 side effects or, (ii) 5 drugs with a total #side-effects >= 5 
df_pres_pat_count = df_pres_pat_count[df_pres_pat_count["count"]>5]
list_pat_matching = list(df_pres_pat_count.index) 

print("#prescriptions: ")
print("df_pres.shape: ",df_pres.shape[0])
print("#")
print("#prescriptions with drugs_common: ")
print("df_pres_drugs_common.shape: ",df_pres_drugs_common.shape[0])
print("#")
print("#patients: ")
print("df_pres[\"SUBJECT_ID\"].unique().shape[0]: ",df_pres["SUBJECT_ID"].unique().shape[0])
print("#")
print("#patients administered (i) drugs with atleast 5 side effects or, (ii) 5 drugs with a total #side-effects >= 5: ")
print("df_pres_pat_count: ",df_pres_pat_count.shape[0])
print("len(list_pat_matching): ",len(list_pat_matching))
print("len(np.unique(list_pat_matching)): ",len(np.unique(list_pat_matching)))
print("#")

# 3b - create a subset of 5K patients from the above list of patients
# list_pat_matching, based on LOS
df_admn_subset = df_admn[df_admn["SUBJECT_ID"].isin(list_pat_matching)]
#ensure there exists a record in admn table for all the pat
list_admin_pat_all = df_admn_subset["SUBJECT_ID"].unique().tolist()
assert len(list(set(list_admin_pat_all)-set(list_pat_matching))) == 0
#
#Find LOS of each patient
df_admn_subset["ADMITTIME"] = pd.to_datetime(df_admn_subset["ADMITTIME"])
df_admn_subset["DISCHTIME"] = pd.to_datetime(df_admn_subset["DISCHTIME"])
df_admn_subset["LOS"] = (df_admn_subset["DISCHTIME"] - df_admn_subset["ADMITTIME"]).astype('timedelta64[h]')
df_admn_subset_stats = df_admn_subset["LOS"].describe()
#
df_temp_gt = df_admn_subset[df_admn_subset["LOS"] >= (df_admn_subset_stats["mean"] + df_admn_subset_stats["std"])]
df_temp_lt = df_admn_subset[df_admn_subset["LOS"] <= (df_admn_subset_stats["mean"])]
#
patients_common_gt = list(df_temp_gt["SUBJECT_ID"].unique())
patients_common_lt = list(df_temp_lt["SUBJECT_ID"].unique())
print("len(patients_common_gt): ",len(patients_common_gt))
print("len(patients_common_lt): ",len(patients_common_lt))
patients_common = list(np.unique(list(random.sample(patients_common_gt, 3000) + random.sample(patients_common_lt, 3000))))
print("len(patients_common): ",len(patients_common)) #5000
print("#")


# 4 - Now create a temp df with "all" prescriptions of the patients
# from patients_common (note: by "all" we mean drugs that were NOT
# in drugs_common too)

df_pres_pat_all = df_pres[df_pres["SUBJECT_ID"].isin(patients_common)]
assert df_pres_pat_all["SUBJECT_ID"].unique().shape[0] == len(patients_common)

list_drugs_pat_all = df_pres_pat_all["in_rxcui"].unique().tolist()
assert len(list_drugs_pat_all) >= len(drugs_common)

drugs_not_common = list(set(list_drugs_pat_all) - set(drugs_common))

print("#")
print("len(list_drugs_pat_all): ",len(list_drugs_pat_all))
print("len(drugs_common): ",len(drugs_common))
print("#")
print("len(drugs_not_common): ",len(drugs_not_common))
print("#")


# 5 - Similarly, for the same set of patients we sampled i.e. patients_common,
# now create a temp df with "all" diagnosis of the patients
# i.e. including diagnosis NOT in dis_common

df_diag_pat_all = df_diag[\
				  df_diag["SUBJECT_ID"].isin(patients_common)]
				  #&\
				  #df_diag["SUBJECT_ID"].isin(patients_given_not_common_drugs)]
#assert df_diag_pat_all["SUBJECT_ID"].unique().shape[0] == len(patients_common)

#TODO: ensure there is atlease a few diagnosis per patient

list_diag_pat_all = df_diag_pat_all["ICD9_CODE"].unique().tolist()
assert len(list_diag_pat_all) >= len(dis_common)

dis_not_common = list(set(list_diag_pat_all) - set(dis_common))

#pick top (by freq of occurence amongst the patients) 100 diseases 
#from dis_not_common using df_diag
df_diag_pat_all_subset = df_diag_pat_all[df_diag_pat_all["ICD9_CODE"].isin(dis_not_common)]
df_diag_pat_all_subset_count = df_diag_pat_all_subset.groupby(["ICD9_CODE"])["SUBJECT_ID"]\
			   						   .agg(['count']) \
			   						   .sort_values(['count'], ascending=False)
#pick top 100
df_diag_pat_all_subset_count = df_diag_pat_all_subset_count[:100]
dis_not_common_top_100 = list(df_diag_pat_all_subset_count.index)

print("#")
print("len(list_diag_pat_all): ",len(list_diag_pat_all))
print("len(dis_common): ",len(dis_common))
print("#")
print("len(dis_not_common): ",len(dis_not_common))
print("len(dis_not_common_top_100): ",len(dis_not_common_top_100))
print("#")

#
#create matrices
#

def remove_nan(L):
	L_out = [x for x in L if pd.notnull(x)]
	return L_out

#patients
p_id_list = remove_nan(patients_common)
p_id_list =  [np.int(x) for x in p_id_list]
#drugs
r_id_list = remove_nan(list(set(drugs_common).union(drugs_not_common)))
r_id_list =  [str(x) for x in r_id_list]
#diseases
d_id_list = remove_nan(list(set(dis_common).union(dis_not_common_top_100)))

#
num_p = len(p_id_list)
num_r = len(r_id_list)
num_d = len(d_id_list)

#
print("Entity sizes: ")
print("len(p_id_list): ",len(p_id_list))
print("len(r_id_list): ",len(r_id_list))
print("len(d_id_list): ",len(d_id_list))
print("#")
print("num_p: ",num_p)
print("num_r: ",num_r)
print("num_d: ",num_d)
print("#")
#
print("###")
print("p_id_list[:10]: ")
print(p_id_list[:10])
print("#")

print("r_id_list[:10]: ")
print(r_id_list[:10])
print("#")

print("d_id_list[:10]: ")
print(d_id_list[:10])
print("#")

# empty matrices
#X_pr: patient x drug
X_pr = np.zeros((num_p,num_r))

#X_rs: drugs x diseases: side-effects
X_rs = np.zeros((num_r,num_d))

#X_pt: patient x diseases: treated
X_pt = np.zeros((num_p,num_d))


print("matrices: ")
print("X_pr.shape: ",X_pr.shape)
print("X_rs.shape: ",X_rs.shape)
print("X_pt.shape: ",X_pt.shape)
print("#")

def get_dict_id_idx_map(p_id_list):
    num_p = len(p_id_list)
    dict_p_id_idx_map = {}
    dict_p_idx_id_map= {}
    for p_idx in np.arange(num_p):
        p_id = p_id_list[p_idx]
        dict_p_id_idx_map[p_id] = p_idx
        dict_p_idx_id_map[p_idx] = p_id
    return dict_p_id_idx_map,dict_p_idx_id_map

#id to idx mapping
dict_p_id_idx_map,dict_p_idx_id_map = get_dict_id_idx_map(p_id_list)
dict_r_id_idx_map,dict_r_idx_id_map = get_dict_id_idx_map(r_id_list)
dict_d_id_idx_map,dict_d_idx_id_map = get_dict_id_idx_map(d_id_list)


def populate_matrix(X_pr, df_pres,\
                    p_name, r_name,\
                    p_id_list, r_id_list,\
                    dict_p_id_idx_map, dict_r_id_idx_map):
    #
    temp_debug_p_list = list(dict_p_id_idx_map.keys())
    temp_debug_r_list = list(dict_r_id_idx_map.keys())
    #
    for row_num,row in df_pres.iterrows():
        if int(row_num)%100000 == 0:
            print("#row: ",row_num)
        p_id = row[p_name] #["SUBJECT_ID"]
        r_id = row[r_name] #["NDC"]
        #
        assert type(p_id) == type(p_id_list[0]), "type(p_id): "+str(type(p_id))+", type(p_id_list[0]): "+str(type(p_id_list[0]))
        assert type(p_id) == type(temp_debug_p_list[0]), "type(p_id): "+str(type(p_id))+", type(temp_debug_p_list[0]): "+str(type(temp_debug_p_list[0]))
        #
        #assert type(r_id) == type(r_id_list[0]), "type(r_id): "+str(type(r_id))+", type(r_id_list[0]): "+str(type(r_id_list[0]))+" | r_id: "+str(r_id)+", r_id_list[0]: "+str(r_id_list[0])
        #assert type(r_id) == type(temp_debug_r_list[0]), "type(r_id): "+str(type(r_id))+", type(temp_debug_r_list[0]): "+str(type(temp_debug_r_list[0]))
        #
        assert p_id in p_id_list, "Not found in p_id_list, pid: "+str(pid)
        assert r_id in r_id_list, "Not found in p_id_list, pid: "+str(pid)
        #
        #if p_id in p_id_list and r_id in r_id_list:
        p_idx = dict_p_id_idx_map[p_id]
        r_idx = dict_r_id_idx_map[r_id]
        X_pr[p_idx,r_idx] += 1
    return X_pr

print("Building matrix X_pr...")
df_pres_final_subset = df_pres[df_pres["SUBJECT_ID"].isin(p_id_list) &\
								df_pres["in_rxcui"].isin(r_id_list)]
#
p_name = "SUBJECT_ID"
r_name = "in_rxcui"
X_pr = populate_matrix( X_pr,\
						df_pres_final_subset,\
						p_name, r_name,\
						p_id_list, r_id_list,\
						dict_p_id_idx_map, dict_r_id_idx_map)

print("X_pr.shape: ",X_pr.shape)
print("X_pr: #entries: ",np.sum(X_pr > 0)) 
print("np.prod(X_pr.shape): ",np.prod(X_pr.shape)) 
print("nonzero percent: ",np.sum(X_pr > 0)/np.prod(X_pr.shape))
print("#")

###################################

print("Building matrix X_rs...")
df_drugs_final_subset = df_drugs[df_drugs["RXCUI"].isin(r_id_list) &\
								 df_drugs["ICD9_CODE"].isin(d_id_list)]
r_name = "RXCUI"
s_name = "ICD9_CODE"
X_rs= populate_matrix(X_rs,\
					  df_drugs_final_subset,\
					  r_name, s_name,\
					  r_id_list, d_id_list,\
					  dict_r_id_idx_map, dict_d_id_idx_map)

print("X_rs.shape: ",X_rs.shape)
print("X_rs: #entries: ",np.sum(X_rs > 0)) 
print("np.prod(X_rs.shape): ",np.prod(X_rs.shape)) 
print("nonzero percent: ",np.sum(X_rs > 0)/np.prod(X_rs.shape))
print("#")

###################################

print("Building matrix X_pt...")
df_diag_final_subset = df_diag[df_diag["SUBJECT_ID"].isin(p_id_list) &\
								df_diag["ICD9_CODE"].isin(d_id_list)]
p_name = "SUBJECT_ID"
t_name = "ICD9_CODE"
X_pt= populate_matrix(X_pt, \
					  df_diag_final_subset, \
					  p_name, t_name, \
					  p_id_list, d_id_list, \
					  dict_p_id_idx_map, dict_d_id_idx_map)

print("X_pt.shape: ",X_pt.shape)
print("X_pt: #entries: ",np.sum(X_pt > 0)) 
print("np.prod(X_pt.shape): ",np.prod(X_pt.shape)) 
print("nonzero percent: ",np.sum(X_pt > 0)/np.prod(X_pt.shape))

###################################
#
# Create ground truth discordance details
# case 1: present in data, absent in KB
# case 2: absent in data, present in KB
#
# patients:
# p_id_list -> patients_common 
#
# drugs:
# r_id_list  -> drugs_common, drugs_not_common
#
# diseases:
# d_id_list -> dis_common, dis_not_common_top_100
#

X_rs_sp = coo_matrix(X_rs)
X_pr_sp = coo_matrix(X_pr)
X_pt_sp = coo_matrix(X_pt)
#
r_list_x_rs = X_rs_sp.row
s_list_x_rs = X_rs_sp.col
val_list_x_rs = X_rs_sp.data
#
p_list_x_pr = X_pr_sp.row
r_list_x_pr = X_pr_sp.col
val_list_x_pr = X_pr_sp.data
#
p_list_x_pt = X_pt_sp.row
t_list_x_pt = X_pt_sp.col
val_list_x_pt = X_pt_sp.data
#

print("#")
print("X_rs_sp.count_nonzero(): ",X_rs_sp.count_nonzero())
print("X_pr_sp.count_nonzero(): ",X_pr_sp.count_nonzero())
print("X_pt_sp.count_nonzero(): ",X_pt_sp.count_nonzero())
print("#")

################
# case 1: full #
################
# case 1: present in data, absent in KB
# Find those pairs of (r,s_d)
# ST
# (r,s_d) not in X_rs
# (p,r) in X_pr
# (p,t_d) in X_pd
#
# step 1 - (p,r) in x_pr and (p,t) in x_pt
df_zip_pr = pd.DataFrame(list(zip(p_list_x_pr, r_list_x_pr)))
df_zip_pr.columns = ["p","r"]
df_zip_pt = pd.DataFrame(list(zip(p_list_x_pt, t_list_x_pt)))
df_zip_pt.columns = ["p","t"]
#
df_zip_prt = pd.merge(df_zip_pr, df_zip_pt, left_on=['p'], right_on = ['p'])
# step 2 - (r,s) NOT in x_rs
df_zip_rs = pd.DataFrame(list(zip(r_list_x_rs, s_list_x_rs)))
df_zip_rs.columns = ["r","s"]


# temp_r_list = df_zip_rs['r'].unique().tolist()
# temp_s_list = df_zip_rs['s'].unique().tolist()
# df_zip_prt_subset = df_zip_prt[~(df_zip_prt["r"].isin(temp_r_list) & \
# 										df_zip_prt["t"].isin(temp_s_list))]

#
list_tup_prd = []
count = 0
count_match1 = 0
check_size = 100000
num_passes = df_zip_prt.shape[0]
s_time = time.time()
for idx,row in df_zip_prt.iterrows():
	count+=1
	#
	p = row["p"]
	r = row["r"]
	t = row["t"]
	if (~(df_zip_rs['r'] == r) & ~(df_zip_rs['s'] == t)).all():
		count_match1+=1
		list_tup_prd.append((p,r,t))
	#
	if count % check_size == 0:
		e_time = time.time()
		print("count: ",count," / ",num_passes,\
			" | %: ",np.round((count/num_passes)*100.0,2),\
			" | took: ",np.round((e_time - s_time)/60.0,2),\
			" | count_match1: ",count_match1)

out_fname = out_dname+"list_tup_prd_case1_"+str(version_name)+".pkl"
print("Persisting: ",out_fname)
pkl.dump(list_tup_prd, open(out_fname,"wb"),protocol=2)

print("len(list_tup_prd): ",len(list_tup_prd))
df_zip_prd = pd.DataFrame(list_tup_prd)
df_zip_prd.columns = ["p","r","d"]
print("df_zip_prd.shape: ",df_zip_prd.shape)

# Since almost 4900 of the 5900 patients have such a case 1 problem
# we subset only the topk of the patients 
# Note: However only 100 of r and s have case 1 problems

# Create a subset with only top k patients by freq
df_zip_prd_count = df_zip_prd.groupby(["p"])["r"].agg(["count"]).sort_values(['count'], ascending=False)
p_list_count_topk = list(df_zip_prd_count[:100].index)
df_zip_prd_subset = df_zip_prd[df_zip_prd["p"].isin(p_list_count_topk)]

#
case_1_list_p_idx = df_zip_prd_subset["p"] #.unique().tolist()
case_1_list_r_idx = df_zip_prd_subset["r"] #.unique().tolist()
case_1_list_d_idx = df_zip_prd_subset["d"] #.unique().tolist()

print("#")
print("len(case_1_list_p_idx): ",len(case_1_list_p_idx))
print("len(case_1_list_r_idx): ",len(case_1_list_r_idx))
print("len(case_1_list_d_idx): ",len(case_1_list_d_idx))
print("#")

# Case 1 - convert gt labels to dict with 
# key: idx
# val: wt = #occurence
dict_case1_gt_idx = {}
dict_case1_gt_idx["p"] = dict(Counter(case_1_list_p_idx))
dict_case1_gt_idx["r"] = dict(Counter(case_1_list_r_idx))
dict_case1_gt_idx["d"] = dict(Counter(case_1_list_d_idx))

print("#")
print("len(dict_case1_gt_idx[p]): ",len(dict_case1_gt_idx["p"]))
print("len(dict_case1_gt_idx[r]): ",len(dict_case1_gt_idx["r"]))
print("len(dict_case1_gt_idx[d] ): ",len(dict_case1_gt_idx["d"] ))
print("#")


p_unknown_idx_list_case1 = list(dict_case1_gt_idx["p"].keys())
r_unknown_idx_list_case1 = list(dict_case1_gt_idx["r"].keys())
d_unknown_idx_list_case1 = list(dict_case1_gt_idx["d"].keys())

print("#")
print("len(p_unknown_idx_list_case1): ",len(p_unknown_idx_list_case1))
print("len(r_unknown_idx_list_case1): ",len(r_unknown_idx_list_case1))
print("len(d_unknown_idx_list_case1): ",len(d_unknown_idx_list_case1))
print("#")

# len(p_unknown_id_list_case1):  100
# len(r_unknown_id_list_case1):  72
# len(d_unknown_id_list_case1):  93

dict_case1_gt_size = {
	"p":len(p_unknown_idx_list_case1),
	"r":len(r_unknown_idx_list_case1),
	"d":len(d_unknown_idx_list_case1)
}


p_unknown_id_list_case1 = []
for temp_idx in p_unknown_idx_list_case1:
	temp_id = dict_p_idx_id_map[temp_idx]
	p_unknown_id_list_case1.append(temp_id)

r_unknown_id_list_case1 = []
for temp_idx in r_unknown_idx_list_case1:
	temp_id = dict_r_idx_id_map[temp_idx]
	r_unknown_id_list_case1.append(temp_id)

d_unknown_id_list_case1 = []
for temp_idx in d_unknown_idx_list_case1:
	temp_id = dict_d_idx_id_map[temp_idx]
	d_unknown_id_list_case1.append(temp_id)

print("#")
print("len(p_unknown_id_list_case1): ",len(p_unknown_id_list_case1))
print("len(r_unknown_id_list_case1): ",len(r_unknown_id_list_case1))
print("len(d_unknown_id_list_case1): ",len(d_unknown_id_list_case1))
print("#")

assert len(p_unknown_id_list_case1) == len(p_unknown_idx_list_case1)
assert len(r_unknown_id_list_case1) == len(r_unknown_idx_list_case1)
assert len(d_unknown_id_list_case1) == len(d_unknown_idx_list_case1)

dict_case1_gt_id = {}
dict_case1_gt_id["p"] = p_unknown_id_list_case1
dict_case1_gt_id["r"] = r_unknown_id_list_case1
dict_case1_gt_id["d"] = d_unknown_id_list_case1

print("###")

################
# case 2: full #
################
# Case 2: absent in data, present in KB
# Find those pairs of (r,s_d)
# ST
# (r,s_d) in X_rs
# (p,r) in X_pr
# (p,t_d) NOT in X_pd
#

# step 1 - (p,r) in x_pr and (r,s_d) in X_rs 
df_zip_pr = pd.DataFrame(list(zip(p_list_x_pr, r_list_x_pr)))
df_zip_pr.columns = ["p","r"]
df_zip_rs = pd.DataFrame(list(zip(r_list_x_rs, s_list_x_rs)))
df_zip_rs.columns = ["r","s"]
#
df_zip_prs = pd.merge(df_zip_pr, df_zip_rs, how="left", left_on=['r'], right_on = ['r']).drop_duplicates()
# step 2 - (p,t_d) NOT in X_pd
df_zip_pt = pd.DataFrame(list(zip(p_list_x_pt, t_list_x_pt)))
df_zip_pt.columns = ["p","t"]

# to speed up the next loop
temp_p_list = df_zip_pt['p'].unique().tolist()
temp_t_list = df_zip_pt['t'].unique().tolist()
df_zip_prs_subset = df_zip_prs[~(df_zip_prs["p"].isin(temp_p_list) &\
								df_zip_prs["s"].isin(temp_t_list))]
	
#
list_tup_prd_case2 = []
count = 0
count_match1 = 0
check_size = 300000
num_passes = df_zip_prs_subset.shape[0]
s_time = time.time()
for idx,row in df_zip_prs_subset.iterrows():
	count+=1
	#
	p = row["p"]
	r = row["r"]
	s = row["s"]
	if (~(df_zip_pt['p'] == p) & ~(df_zip_pt['t'] == s)).all():
		count_match1+=1
		list_tup_prd_case2.append((p,r,s))
	#
	if count % check_size == 0:
		e_time = time.time()
		print("count: ",count," / ",num_passes,\
			" | %: ",np.round((count/num_passes)*100.0,2),\
			" | took: ",np.round((e_time - s_time)/60.0,2),\
			" | count_match1: ",count_match1)


out_fname = out_dname+"list_tup_prd_case2_"+str(version_name)+".pkl"
print("Persisting: ",out_fname)
pkl.dump(list_tup_prd_case2, open(out_fname,"wb"),protocol=2)

print("len(list_tup_prd_case2): ",len(list_tup_prd_case2))
df_zip_prd_case2 = pd.DataFrame(list_tup_prd_case2)
df_zip_prd_case2.columns = ["p","r","d"]
print("df_zip_prd_case2.shape: ",df_zip_prd_case2.shape)

# # Since almost 600 of the 1400 (and 48 patients) have a case 2 problem
# # we subset only the topk of the 600 diseases (and the corresponding patients)

# Create a subset with only top k disease by freq
df_zip_prd_case2_count = df_zip_prd_case2.groupby(["d"])["p"].agg(["count"]).sort_values(['count'], ascending=False)
d_list_count_topk_case2 = list(df_zip_prd_case2_count.head(100).index)
df_zip_prd_case2_subset = df_zip_prd_case2[df_zip_prd_case2["d"].isin(d_list_count_topk_case2)]

#df_zip_prd_case2_subset = df_zip_prd_case2
#
case_2_list_p_idx = df_zip_prd_case2_subset["p"] #.unique().tolist()
case_2_list_r_idx = df_zip_prd_case2_subset["r"] #.unique().tolist()
case_2_list_d_idx = df_zip_prd_case2_subset["d"] #.unique().tolist()


print("#")
print("len(case_2_list_p_idx): ",len(case_2_list_p_idx))
print("len(case_2_list_r_idx): ",len(case_2_list_r_idx))
print("len(case_2_list_d_idx): ",len(case_2_list_d_idx))
print("#")


# Case 2 - convert gt labels to dict with 
# key: idx
# val: wt = #occurence
dict_case2_gt_idx = {}
dict_case2_gt_idx["p"] = dict(Counter(case_2_list_p_idx))
dict_case2_gt_idx["r"] = dict(Counter(case_2_list_r_idx))
dict_case2_gt_idx["d"] = dict(Counter(case_2_list_d_idx))

print("#")
print("len(dict_case2_gt_idx[p]): ",len(dict_case2_gt_idx["p"]))
print("len(dict_case2_gt_idx[r]): ",len(dict_case2_gt_idx["r"]))
print("len(dict_case2_gt_idx[d]): ",len(dict_case2_gt_idx["d"]))
print("#")

p_unknown_idx_list_case2 = list(dict_case2_gt_idx["p"].keys())
r_unknown_idx_list_case2 = list(dict_case2_gt_idx["r"].keys())
d_unknown_idx_list_case2 = list(dict_case2_gt_idx["d"].keys())

print("#")
print("len(p_unknown_idx_list_case2): ",len(p_unknown_idx_list_case2))
print("len(r_unknown_idx_list_case2): ",len(r_unknown_idx_list_case2))
print("len(d_unknown_idx_list_case2): ",len(d_unknown_idx_list_case2))
print("#")


p_unknown_id_list_case2 = []
for temp_idx in p_unknown_idx_list_case2:
	temp_id = dict_p_idx_id_map[temp_idx]
	p_unknown_id_list_case2.append(temp_id)

r_unknown_id_list_case2 = []
for temp_idx in r_unknown_idx_list_case2:
	temp_id = dict_r_idx_id_map[temp_idx]
	r_unknown_id_list_case2.append(temp_id)

d_unknown_id_list_case2 = []
for temp_idx in d_unknown_idx_list_case2:
	temp_id = dict_d_idx_id_map[temp_idx]
	d_unknown_id_list_case2.append(temp_id)

assert len(p_unknown_id_list_case2) == len(p_unknown_idx_list_case2)
assert len(r_unknown_id_list_case2) == len(r_unknown_idx_list_case2)
assert len(d_unknown_id_list_case2) == len(d_unknown_idx_list_case2)

print("#")
print("len(p_unknown_id_list_case2): ",len(p_unknown_id_list_case2))
print("len(r_unknown_id_list_case2): ",len(r_unknown_id_list_case2))
print("len(d_unknown_id_list_case2): ",len(d_unknown_id_list_case2))
print("#")

# #
# len(p_unknown_id_list_case2):  48
# len(r_unknown_id_list_case2):  86
# len(d_unknown_id_list_case2):  100
#
dict_case2_gt_size = {
	"p":len(p_unknown_id_list_case2),
	"r":len(r_unknown_id_list_case2),
	"d":len(d_unknown_id_list_case2)
}

dict_case2_gt_id = {}
dict_case2_gt_id["p"] = p_unknown_id_list_case2
dict_case2_gt_id["r"] = r_unknown_id_list_case2
dict_case2_gt_id["d"] = d_unknown_id_list_case2


# create overall data dict and persist 

out_dict = {}

out_dict["matrices_data"] = {
                 "mat_pat_dis_treat":X_pt,
                 "mat_pat_drugs":X_pr,
                 "mat_drugs_dis_side":X_rs
                 }

out_dict["metadata"] = {
    "dict_p_id_idx_map":dict_p_id_idx_map,
    "dict_r_id_idx_map":dict_r_id_idx_map,
    "dict_d_id_idx_map":dict_d_id_idx_map,
    "dict_p_idx_id_map":dict_p_idx_id_map,
    "dict_r_idx_id_map":dict_r_idx_id_map,
    "dict_d_idx_id_map":dict_d_idx_id_map,
    }

out_dict["gt_case1"] = {
		"r_unknown_id_list": r_unknown_id_list_case1,
	    "s_unknown_id_list": d_unknown_id_list_case1,
	    "p_unknown_id_list": p_unknown_id_list_case1,
	    "t_unknown_id_list": d_unknown_idx_list_case1,
	    "r_unknown_idx_list": r_unknown_idx_list_case1,
	    "s_unknown_idx_list": d_unknown_idx_list_case1,
	    "p_unknown_idx_list": p_unknown_idx_list_case1,
	    "t_unknown_idx_list": d_unknown_idx_list_case1,
	    "dict_gt_idx_wt": dict_case1_gt_idx,
	    "dict_gt_entity_size": dict_case1_gt_size
	    }

out_dict["gt_case2"] = {
		"r_unknown_id_list": r_unknown_id_list_case2,
	    "s_unknown_id_list": d_unknown_id_list_case2,
	    "p_unknown_id_list": p_unknown_id_list_case2,
	    "t_unknown_id_list": d_unknown_idx_list_case2,
	    "r_unknown_idx_list": r_unknown_idx_list_case2,
	    "s_unknown_idx_list": d_unknown_idx_list_case2,
	    "p_unknown_idx_list": p_unknown_idx_list_case2,
	    "t_unknown_idx_list": d_unknown_idx_list_case2,
	    "dict_gt_idx_wt": dict_case2_gt_idx,
	    "dict_gt_entity_size": dict_case2_gt_size
	    }

out_fname = out_dname+"dict_nsides_mimic_data_v"+str(version_name)+".pkl"
print("Persisting: ",out_fname)
pkl.dump(out_dict,open(out_fname,"wb"),protocol=2)


# dict_case1_gt_size (top 100 p)
# {'p': 100, 'r': 72, 'd': 93}
# dict_case2_gt_size (top 100 d)
# {'p': 48, 'r': 86, 'd': 100}

#
#create matrices - case 2 - partial matrices
#

# def remove_nan(L):
#     L_out = [x for x in L if pd.notnull(x)]
#     return L_out

#patients - no change
#p_id_list = remove_nan(patients_common)
#p_id_list =  [np.int(x) for x in p_id_list]
p_id_list_part = p_id_list

#drugs - only common
#r_id_list = remove_nan(list(set(drugs_common).union(drugs_not_common)))
r_id_list_part = remove_nan(list(set(drugs_common)))
r_id_list_part =  [str(x) for x in r_id_list_part]

#diseases - only common
#d_id_list = remove_nan(list(set(dis_common).union(dis_not_common_top_100)))
d_id_list_part = remove_nan(list(set(dis_common)))

#
num_p_part = len(p_id_list_part)
num_r_part = len(r_id_list_part)
num_d_part = len(d_id_list_part)

#
print("Entity sizes: ")
print("len(p_id_list_part): ",len(p_id_list_part))
print("len(r_id_list_part): ",len(r_id_list_part))
print("len(d_id_list_part): ",len(d_id_list_part))
print("#")
print("num_p_part: ",num_p_part)
print("num_r_part: ",num_r_part)
print("num_d_part: ",num_d_part)
print("#")
#
print("###")
print("p_id_list_part[:10]: ")
print(p_id_list_part[:10])
print("#")

print("r_id_list_part[:10]: ")
print(r_id_list_part[:10])
print("#")

print("d_id_list_part[:10]: ")
print(d_id_list_part[:10])
print("#")

# empty matrices
#X_pr: patient x drug
X_pr_part = np.zeros((num_p_part,num_r_part))

#X_rs: drugs x diseases: side-effects
X_rs_part = np.zeros((num_r_part,num_d_part))

#X_pt: patient x diseases: treated
X_pt_part = np.zeros((num_p_part,num_d_part))


print("matrices: ")
print("X_pr_part.shape: ",X_pr_part.shape)
print("X_rs_part.shape: ",X_rs_part.shape)
print("X_pt_part.shape: ",X_pt_part.shape)
print("#")

# def get_dict_id_idx_map(p_id_list):
#     num_p = len(p_id_list)
#     dict_p_id_idx_map = {}
#     dict_p_idx_id_map= {}
#     for p_idx in np.arange(num_p):
#         p_id = p_id_list[p_idx]
#         dict_p_id_idx_map[p_id] = p_idx
#         dict_p_idx_id_map[p_idx] = p_id
#     return dict_p_id_idx_map,dict_p_idx_id_map

#id to idx mapping
dict_p_id_idx_map_part,dict_p_idx_id_map_part = get_dict_id_idx_map(p_id_list_part)
dict_r_id_idx_map_part,dict_r_idx_id_map_part = get_dict_id_idx_map(r_id_list_part)
dict_d_id_idx_map_part,dict_d_idx_id_map_part = get_dict_id_idx_map(d_id_list_part)


# def populate_matrix(X_pr, df_pres,\
#                     p_name, r_name,\
#                     p_id_list, r_id_list,\
#                     dict_p_id_idx_map, dict_r_id_idx_map):
#     #
#     temp_debug_p_list = list(dict_p_id_idx_map.keys())
#     temp_debug_r_list = list(dict_r_id_idx_map.keys())
#     #
#     for row_num,row in df_pres.iterrows():
#         if int(row_num)%100000 == 0:
#             print("#row: ",row_num)
#         p_id = row[p_name] #["SUBJECT_ID"]
#         r_id = row[r_name] #["NDC"]
#         #
#         assert type(p_id) == type(p_id_list[0]), "type(p_id): "+str(type(p_id))+", type(p_id_list[0]): "+str(type(p_id_list[0]))
#         assert type(p_id) == type(temp_debug_p_list[0]), "type(p_id): "+str(type(p_id))+", type(temp_debug_p_list[0]): "+str(type(temp_debug_p_list[0]))
#         #
#         #assert type(r_id) == type(r_id_list[0]), "type(r_id): "+str(type(r_id))+", type(r_id_list[0]): "+str(type(r_id_list[0]))+" | r_id: "+str(r_id)+", r_id_list[0]: "+str(r_id_list[0])
#         #assert type(r_id) == type(temp_debug_r_list[0]), "type(r_id): "+str(type(r_id))+", type(temp_debug_r_list[0]): "+str(type(temp_debug_r_list[0]))
#         #
#         assert p_id in p_id_list, "Not found in p_id_list, pid: "+str(pid)
#         assert r_id in r_id_list, "Not found in p_id_list, pid: "+str(pid)
#         #
#         #if p_id in p_id_list and r_id in r_id_list:
#         p_idx = dict_p_id_idx_map[p_id]
#         r_idx = dict_r_id_idx_map[r_id]
#         X_pr[p_idx,r_idx] += 1
#     return X_pr

print("Building matrix X_pr_part...")
df_pres_final_subset_part = df_pres[df_pres["SUBJECT_ID"].isin(p_id_list_part) &\
                                df_pres["in_rxcui"].isin(r_id_list_part)]
#
p_name = "SUBJECT_ID"
r_name = "in_rxcui"
X_pr_part = populate_matrix(X_pr_part,\
                        df_pres_final_subset_part,\
                        p_name, r_name,\
                        p_id_list_part, r_id_list_part,\
                        dict_p_id_idx_map_part, dict_r_id_idx_map_part)

print("X_pr_part.shape: ",X_pr_part.shape)
print("X_pr_part: #entries: ",np.sum(X_pr_part > 0)) 
print("np.prod(X_pr_part.shape): ",np.prod(X_pr_part.shape)) 
print("nonzero percent: ",np.sum(X_pr_part > 0)/np.prod(X_pr_part.shape))
print("#")

# X_pr_part.shape:  (5891, 596)
# X_pr_part: #entries:  129724
# np.prod(X_pr_part.shape):  3511036
# nonzero percent:  0.03694749925662967

###################################

print("Building matrix X_rs_part...")
df_drugs_final_subset_part = df_drugs[df_drugs["RXCUI"].isin(r_id_list_part) &\
                                 df_drugs["ICD9_CODE"].isin(d_id_list_part)]
r_name = "RXCUI"
s_name = "ICD9_CODE"
X_rs_part = populate_matrix(X_rs_part,\
                      df_drugs_final_subset_part,\
                      r_name, s_name,\
                      r_id_list_part, d_id_list_part,\
                      dict_r_id_idx_map_part, dict_d_id_idx_map_part)

print("X_rs_part.shape: ",X_rs_part.shape)
print("X_rs_part: #entries: ",np.sum(X_rs_part > 0)) 
print("np.prod(X_rs_part.shape): ",np.prod(X_rs_part.shape)) 
print("nonzero percent: ",np.sum(X_rs_part > 0)/np.prod(X_rs_part.shape))
print("#")

# X_rs_part.shape:  (596, 1321)
# X_rs_part: #entries:  347773
# np.prod(X_rs_part.shape):  787316
# nonzero percent:  0.4417197160987456


###################################

print("Building matrix X_pt_part...")
df_diag_final_subset_part = df_diag[df_diag["SUBJECT_ID"].isin(p_id_list_part) &\
                                df_diag["ICD9_CODE"].isin(d_id_list_part)]
p_name = "SUBJECT_ID"
t_name = "ICD9_CODE"
X_pt_part = populate_matrix(X_pt_part, \
                      df_diag_final_subset_part, \
                      p_name, t_name, \
                      p_id_list_part, d_id_list_part, \
                      dict_p_id_idx_map_part, dict_d_id_idx_map_part)

print("X_pt_part.shape: ",X_pt_part.shape)
print("X_pt_part: #entries: ",np.sum(X_pt_part > 0)) 
print("np.prod(X_pt_part.shape): ",np.prod(X_pt_part.shape)) 
print("nonzero percent: ",np.sum(X_pt_part > 0)/np.prod(X_pt_part.shape))

# X_pt_part.shape:  (5891, 1321)
# X_pt_part: #entries:  29148
# np.prod(X_pt_part.shape):  7782011
# nonzero percent:  0.003745561397947137


###################################
#
# Create ground truth discordance details
# case 1: present in data, absent in KB - only parital/common entity data
#
# patients:
# p_id_list -> patients_common 
#
# drugs:
# r_id_list  -> drugs_common
#
# diseases:
# d_id_list -> dis_common
#

X_rs_sp_part = coo_matrix(X_rs_part)
X_pr_sp_part = coo_matrix(X_pr_part)
X_pt_sp_part = coo_matrix(X_pt_part)
#
r_list_x_rs_part = X_rs_sp_part.row
s_list_x_rs_part = X_rs_sp_part.col
val_list_x_rs_part = X_rs_sp_part.data
#
p_list_x_pr_part = X_pr_sp_part.row
r_list_x_pr_part = X_pr_sp_part.col
val_list_x_pr_part = X_pr_sp_part.data
#
p_list_x_pt_part = X_pt_sp_part.row
t_list_x_pt_part = X_pt_sp_part.col
val_list_x_pt_part = X_pt_sp_part.data
#

print("#")
print("X_rs_sp_part.count_nonzero(): ",X_rs_sp_part.count_nonzero())
print("X_pr_sp_part.count_nonzero(): ",X_pr_sp_part.count_nonzero())
print("X_pt_sp_part.count_nonzero(): ",X_pt_sp_part.count_nonzero())
print("#")


###################
# case 1: partial #
###################
s_percent = 5
r_percent = 10
num_r_unknown = int(num_r_part * ((r_percent/100.0)))
# Select a random set of drugs with atleast one entry in X_rs_part
print("#")
print("Creating case1 partial - dataset with hidden (r,s) entries")
print("#")
print("Selecting r:")
# random r selection - the set of drugs r which exhibit "unknown" side effects
# It becomes "unknown" because it the associated side effects will be hidden
#
r_unknown_idx_list = random.sample(list(np.unique(r_list_x_rs_part)),num_r_unknown)
print("len(r_unknown_idx_list): ",len(r_unknown_idx_list))
print("#")
print("r_unknown_idx_list[:10]: ")
print(r_unknown_idx_list[:10])
print("###")

# X_rs_part
# For each r selected in the previous find all the associated side effects s
# Select s_percent of the associated s
# Hide the entries in X_rs_part corresponding to the selecting r and s indexes
print("Selecting associated s:")
print("#")
print("s_percent: ",s_percent)
print("#")
print("X_rs_part: #entries before masking: ",np.sum(X_rs_part > 0))
temp_bef_count = np.sum(X_rs_part > 0)
#
list_rs_tup_hidden = []
s_unknown_idx_list = []
for r_unknown_idx in r_unknown_idx_list:
    temp_s_unknown_idx_list = []
    ij_idx_list_tup = np.nonzero(X_rs_part[r_unknown_idx,:])
    j_idx_list = ij_idx_list_tup[0]
    num_s_unknown = int(np.ceil(len(j_idx_list) * (s_percent/100.0))) 
    for count_idx in np.arange(num_s_unknown):
        s_unknown_idx = j_idx_list[count_idx]
        assert X_rs_part[r_unknown_idx,s_unknown_idx] > 0
        X_rs_part[r_unknown_idx,s_unknown_idx] = 0
        temp_s_unknown_idx_list.append(s_unknown_idx)
        list_rs_tup_hidden.append((r_unknown_idx,s_unknown_idx))
    s_unknown_idx_list.extend(temp_s_unknown_idx_list)

print("X_rs_part: #entries after masking: ",np.sum(X_rs_part > 0))
temp_aft_count = np.sum(X_rs_part > 0)
print("len(s_unknown_idx_list): ",len(s_unknown_idx_list))
assert temp_aft_count == (temp_bef_count-len(s_unknown_idx_list))
print("#")
print("len(list_rs_tup_hidden): ",len(list_rs_tup_hidden))
print("#")
print("s_unknown_idx_list[:10]: ")
print(s_unknown_idx_list[:10])
print("#")
print("list_rs_tup_hidden[:10]: ")
print(list_rs_tup_hidden[:10])
print("#")

# Find the patients {p_i} associated with the drugs r_i, 
# whose side effects s_j were hidden in the previous step
# ST
# X_pr(p_i,r_i) is present and
# X_pr(p_i,s_j) is present and
print("Finding associated (p,r) and (p,t) entries:")
print("#")
p_unknown_idx_list = []
t_unknown_idx_list = []
list_pr_tup_asso_hidden = []
list_pt_tup_asso_hidden = []
for rs_tup in list_rs_tup_hidden:
    r_unknown_idx = rs_tup[0]
    s_unknown_idx = rs_tup[1]
    #find p_*,cur_r_unknown
    temp_ij_idx_list_tup = np.nonzero(X_pr_part[:,r_unknown_idx])
    p_idx_x_pr_nz_list = list(temp_ij_idx_list_tup[0])
    #find p_*,cur_s_unknown
    temp_ij_idx_list_tup = np.nonzero(X_pt_part[:,s_unknown_idx])
    p_idx_x_pt_nz_list = list(temp_ij_idx_list_tup[0])
    #
    p_idx_common_list = list(set(p_idx_x_pr_nz_list).intersection(set(p_idx_x_pt_nz_list)))
    if len(p_idx_common_list):
        p_unknown_idx_list.extend(p_idx_common_list)
        t_unknown_idx_list.append(s_unknown_idx)
        #
        for temp_p_idx in p_unknown_idx_list:
            list_pr_tup_asso_hidden.append((temp_p_idx,r_unknown_idx))
            list_pt_tup_asso_hidden.append((temp_p_idx,s_unknown_idx))

print("#")
print("len(p_unknown_idx_list): ",len(p_unknown_idx_list))
print("len(t_unknown_idx_list): ",len(t_unknown_idx_list))
print("#")
print("#")
print("len(list_pr_tup_asso_hidden): ",len(list_pr_tup_asso_hidden))
print("len(list_pt_tup_asso_hidden): ",len(list_pt_tup_asso_hidden))
print("#")    



#
#find weight of each of the hidden entity r,s or entity associated with the hidden i.e. p,t
dict_case1_part_gt_idx = {}
dict_case1_part_gt_idx["p"] = dict(Counter(p_unknown_idx_list))
dict_case1_part_gt_idx["r"] = dict(Counter(r_unknown_idx_list))
dict_case1_part_gt_idx["s"] = dict(Counter(s_unknown_idx_list))
dict_case1_part_gt_idx["t"] = dict(Counter(t_unknown_idx_list))

print("#")
print("Before uniq: ")
print("len(p_unknown_idx_list): ",len(p_unknown_idx_list))
print("len(r_unknown_idx_list): ",len(r_unknown_idx_list))
print("len(s_unknown_idx_list): ",len(s_unknown_idx_list))
print("len(t_unknown_idx_list): ",len(t_unknown_idx_list))
print("#")

p_unknown_idx_list = list(np.unique(p_unknown_idx_list))
r_unknown_idx_list = list(np.unique(r_unknown_idx_list))
s_unknown_idx_list = list(np.unique(s_unknown_idx_list))
t_unknown_idx_list = list(np.unique(t_unknown_idx_list))

print("#")
print("After uniq: ")
print("len(p_unknown_idx_list): ",len(p_unknown_idx_list))
print("len(r_unknown_idx_list): ",len(r_unknown_idx_list))
print("len(s_unknown_idx_list): ",len(s_unknown_idx_list))
print("len(t_unknown_idx_list): ",len(t_unknown_idx_list))
print("#")

#find the idx's corresponding to the unknown idx

p_unknown_id_list = []
for cur_idx in p_unknown_idx_list:
    p_unknown_id_list.append(dict_p_idx_id_map_part[cur_idx])

r_unknown_id_list = []
for cur_idx in r_unknown_idx_list:
    r_unknown_id_list.append(dict_r_idx_id_map_part[cur_idx])

s_unknown_id_list = []
for cur_idx in s_unknown_idx_list:
    s_unknown_id_list.append(dict_d_idx_id_map_part[cur_idx])

t_unknown_id_list = []
for cur_idx in t_unknown_idx_list:
    t_unknown_id_list.append(dict_d_idx_id_map_part[cur_idx])


dict_case1_part_gt_size = {
    "p":len(p_unknown_id_list),
    "r":len(r_unknown_id_list),
    "s":len(s_unknown_id_list),
    "t":len(t_unknown_id_list)
}

#

out_dict_part = {}

out_dict_part["matrices_data"] = {
                 "mat_pat_dis_treat":X_pt_part,
                 "mat_pat_drugs":X_pr_part,
                 "mat_drugs_dis_side":X_rs_part
                 }

out_dict_part["metadata"] = {
    "dict_p_id_idx_map":dict_p_id_idx_map_part,
    "dict_r_id_idx_map":dict_r_id_idx_map_part,
    "dict_d_id_idx_map":dict_d_id_idx_map_part,
    "dict_p_idx_id_map":dict_p_idx_id_map_part,
    "dict_r_idx_id_map":dict_r_idx_id_map_part,
    "dict_d_idx_id_map":dict_d_idx_id_map_part,
    }

out_dict_part["gt_case1_part"] = {
        "r_unknown_id_list": r_unknown_id_list,
        "s_unknown_id_list": s_unknown_id_list,
        "p_unknown_id_list": p_unknown_id_list,
        "t_unknown_id_list": t_unknown_idx_list,
        "r_unknown_idx_list": r_unknown_idx_list,
        "s_unknown_idx_list": s_unknown_idx_list,
        "p_unknown_idx_list": p_unknown_idx_list,
        "t_unknown_idx_list": t_unknown_idx_list,
        "dict_gt_idx_wt": dict_case1_part_gt_idx,
        "dict_gt_entity_size": dict_case1_part_gt_size,
        "list_rs_tup_hidden": list_rs_tup_hidden,
        "list_pr_tup_asso_hidden": list_pr_tup_asso_hidden,
        "list_pt_tup_asso_hidden": list_pt_tup_asso_hidden
        }

out_fname_part = out_dname+"dict_nsides_mimic_data_v"+str(version_name)+"_case1_part.pkl"
print("Persisting: ",out_fname_part)
pkl.dump(out_dict_part,open(out_fname_part,"wb"),protocol=2)

