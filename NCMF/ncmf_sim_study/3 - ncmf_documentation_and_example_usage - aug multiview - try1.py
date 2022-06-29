#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("..")
#
import pprint
import numpy as np
import pickle as pkl
import time
import itertools
import os
import pprint
#
from src_sim.ncmf import ncmf
#
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
pp = pprint.PrettyPrinter()

from IPython import get_ipython

# ## NCMF
# Example of running the "NCMF" module

# #### *User inputs*

# In[2]:


sample_no = 1
data_dir = f"./ncmf_sim_data"
#list_dataset_names = ["dt1"]
list_dataset_names = ['dt1', 'ds1', 'ds2', 'ds3', 'dn1', 'dn2', 'dn3']
#list_dataset_names = ["ds1","ds2","ds3"]
#list_dataset_names = ["dn1","dn2","dn3"]


# In[ ]:


for dataset_name in list_dataset_names:
    print("#")
    os.system(' mkdir -p {data_dir}/{dataset_name}/{sample_no}')
    #
    # Setting hyperparameters
    num_epochs = 300
    batch_size = 2048
    weight_decay = 1e-3
    learning_rate = 1e-5
    convergence_threshold = -1e-3
    entity_matrices = ["X0","X1","X2","X3","X4"]
    matrix_types = {
        "real": ["X0","X1","X3"],
        "binary": ["X2","X4"]
    }
    ncmf_model = ncmf(sample_no, data_dir, dataset_name,                      matrix_types, num_epochs, learning_rate,                       weight_decay, convergence_threshold, batch_size,                       batch_size, entity_matrices, autoencoder_act_f = "tanh", reconstructor_act_f = "tanh")
    ncmf_model.fit()
    print("#")


# In[ ]:


print("done")


# In[ ]:




