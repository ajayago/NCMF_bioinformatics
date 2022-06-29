
# coding: utf-8

# In[1]:


import pickle as pkl
import os
#
import pandas as pd
import numpy as np
import random


# In[2]:


import sys
sys.path.append("..")


# In[3]:


import pprint
import numpy as np
import pickle as pkl
import time
import itertools
import os


# In[4]:


from src.dcmf import dcmf


# In[5]:


base_dname = "./../../ncmf_sim_data/"
in_dir = base_dname + "cmf/"
out_dir_base = base_dname + "dcmf/out/"


# In[6]:


for dataset_name in ['dt1', 'ds1', 'ds2', 'ds3', 'dn1', 'dn2', 'dn3']:
    print("dataset_name: ",dataset_name)
    print("#")
    #
    fname = in_dir + dataset_name + "/0.csv"
    X0 = pd.read_csv(fname,header=None).values
    #
    fname = in_dir + dataset_name + "/1.csv"
    X1 = pd.read_csv(fname,header=None).values
    #
    fname = in_dir + dataset_name + "/2.csv"
    X2 = pd.read_csv(fname,header=None).values
    #
    fname = in_dir + dataset_name + "/3.csv"
    X3 = pd.read_csv(fname,header=None).values
    #
    fname = in_dir + dataset_name + "/4.csv"
    X4 = pd.read_csv(fname,header=None).values
    #
    out_dir = out_dir_base + dataset_name + "/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #
    print("X0.shape: ",X0.shape)
    print("X1.shape: ",X1.shape)
    print("X2.shape: ",X2.shape)
    print("X3.shape: ",X3.shape)
    print("X4.shape: ",X4.shape)
    #
    G = {"e0":["X0","X1","X2"],         "e1":["X0","X3"],         "e2":["X1","X4"],         "e3":["X2"],         "e4":["X3"],         "e5":["X4"]}
    #
    X_data = {
        "X0":X0,
        "X1":X1,
        "X2":X2,
        "X3":X3,
        "X4":X4}
    #
    X_meta = {"X0":["e0","e1"],         "X1":["e0","e2"],         "X2":["e0","e3"],         "X3":["e4","e1"],         "X4":["e5","e2"]}
    #
    X_val = {}
    #
    kf = 0.0005
    k = 100
    e_actf = "tanh"
    d_actf = "tanh"
    is_linear_last_enc_layer = False
    is_linear_last_dec_layer = False
    num_chunks = 10
    #
    learning_rate = 0.0001
    weight_decay = 0.01
    max_epochs = 100
    convg_thres = -0.1
    #
    is_pretrain=False
    pretrain_thres= 0.1
    max_pretrain_epochs = 2
    #
    val_metric = "auc"
    is_val_transpose = True
    at_k = 10
    #
    is_gpu = True
    gpu_ids = "1"
    #
    num_folds = 1
    #
    dcmf_model = dcmf(G, X_data, X_meta,                num_chunks=num_chunks,k=k, kf=kf, e_actf=e_actf, d_actf=d_actf,                learning_rate=learning_rate, weight_decay=weight_decay, convg_thres=convg_thres, max_epochs=max_epochs,                is_gpu=is_gpu,gpu_ids=gpu_ids,is_pretrain=is_pretrain, pretrain_thres=pretrain_thres,                max_pretrain_epochs=max_pretrain_epochs,X_val=X_val,val_metric=val_metric,                is_val_transpose=is_val_transpose, at_k=at_k,                is_linear_last_enc_layer=is_linear_last_enc_layer,is_linear_last_dec_layer=is_linear_last_dec_layer,num_val_sets=num_folds)
    #
    dcmf_model.fit()
    #
    dict_out = dcmf_model.out_dict_X_prime["1"]
    dict_out_np = {}
    for cur_mat_id in dict_out:
        cur_mat_tensor = dict_out[cur_mat_id]
        cur_mat_np = cur_mat_tensor.cpu().detach().numpy()
        dict_out_np[cur_mat_id] = cur_mat_np
    #
    fname_out = out_dir + "dict_out_dcmf.pkl"
    pkl.dump(dict_out_np,open(fname_out,"wb"))


# In[ ]:


print("done")

