import numpy as np
np.random.seed(0)
import pandas as pd
import os
from datetime import datetime
import time
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from math import pi
from numpy import sin, cos, linspace
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.sparse import coo_matrix
import math

class SyntheticDataGeneratorBioinf(object):

    def __init__(self,\
        k, G, dict_entity_size, dict_mat_dtype,\
        list_mod_mat_id, sparsity_level, noise_level):
        # k = 100
        # G = {"1":["1","2"],"2":["1","3"],"3":["1","4"],"4":["6","2"],"5":["5","3"]}
        # dict_entity_size = {'1':20, '2':50, '3':30, '4':40, '5':10, '6':60}
        # dict_mat_dtype = {'1':"real", '2':"real", '3':"binary", '4':"real", '5':"real", '6':"real"}
        # #key: matrix index, 
        # #value: list; where,
        # #list[0] - row entity index, list[1] - col
        # #
        # list_mod_mat_id = ["1","3"] # we modify the sparsity and the noise level only to these matrices
        # sparsity_level = 0
        # noise_level = 1
        # # 0 - no modification to the sparsity/noise
        # # 1 to 3 - increase in the number => increase in noise/sparsity level i.e. number of zeros
        self.k = k
        self.G = G
        self.dict_entity_size = dict_entity_size
        self.dict_mat_dtype = dict_mat_dtype
        self.list_mod_mat_id = list_mod_mat_id
        self.sparsity_level = sparsity_level
        self.noise_level = noise_level
        ### val
        list_matrices = list(G.keys())
        num_matrices = len(list_matrices)
        num_uniq_matrices = len(np.unique(list(G.keys())))
        assert num_matrices == num_uniq_matrices
        #
        list_all_entities = []
        for cur_list in G.values():
            list_all_entities.extend(cur_list)
        list_uniq_entities = list(np.unique(list_all_entities))
        num_entities = len(list_uniq_entities)
        #
        list_uniq_entities_2 = list(dict_entity_size.keys())
        assert set(list_uniq_entities_2) == set(list_uniq_entities)
        #
        assert np.all(np.array(list(dict_entity_size.values())) > 0) == True
        #
        list_dtypes = list(dict_mat_dtype.values())
        for cur_dtype in list_dtypes:
            assert cur_dtype in ["real","binary"]
        #
        assert sparsity_level in [0,1,2,3]
        assert noise_level in [0,1,2,3]

    def __get_random_num(self): #between 0 and 1
        #return np.random.random(size=1)[0]
        return np.random.choice(np.arange(5,15))

    def __get_random_num2(self): #between 0 and 1
        #return np.random.random(size=1)[0]
        return np.random.choice(np.arange(15,45))

    def __get_random_matrix(self, num_rows, num_cols):
        #Return random floats in the half-open interval [0.0, 1.0)
        return np.random.random((num_rows,num_cols))

    def __get_normal_matrix(self, num_rows, num_cols, mu, sigma):
        return np.random.normal(mu, sigma, size=(num_rows,num_cols))

    def __get_X_flipped(self, X, num_flips):
    #    print("num_flips: ",num_flips)
    #     #
    #     ax1 = plt.axes()
    #     sns.heatmap(pd.DataFrame(X),ax = ax1)
    #     ax1.set_title('bef')
    #     plt.show()
        #
        ones_ij_list = np.argwhere(X == 1)
        zeros_ij_list = np.argwhere(X == 0)
        #
        num_ones_bef = len(ones_ij_list)
        num_zeros_bef = len(zeros_ij_list)
        #
        #ones to zeros
        list_sample_idx_ones = random.sample(list(np.arange(ones_ij_list.shape[0])), num_flips)
        list_sample_ij_ones = ones_ij_list[list_sample_idx_ones]
        for ij_tup in list_sample_ij_ones:
            i = ij_tup[0]
            j = ij_tup[1]
            X[i,j] = 0
        #
        #zeros to ones
        list_sample_idx_zeros = random.sample(list(np.arange(zeros_ij_list.shape[0])), num_flips)
        list_sample_ij_zeros = zeros_ij_list[list_sample_idx_zeros]
        for ij_tup in list_sample_ij_zeros:
            i = ij_tup[0]
            j = ij_tup[1]
            X[i,j] = 1
        #
        ones_ij_list_aft = np.argwhere(X == 1)
        zeros_ij_list_aft = np.argwhere(X == 0)
        #
        assert len(ones_ij_list_aft) == num_ones_bef
        assert len(zeros_ij_list_aft) == num_zeros_bef
    #     #
    #     ax1 = plt.axes()
    #     sns.heatmap(pd.DataFrame(X),ax = ax1)
    #     ax1.set_title('aft')
    #     plt.show()
        #
        return X

    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def f(self, x, sparsity_level, noise_level):
        #x_non_linear_transf = (x**3 - 3*x)
        x_non_linear_transf = x
        #return x_non_linear_transf
        scaler = StandardScaler()
        mat_scaled = scaler.fit_transform(x_non_linear_transf)
        mat_scaled_sparse = mat_scaled.copy()
        #
        if sparsity_level == 0:
            thres = -0.6745 # 0.25 percentile
        elif sparsity_level in [1]:
            thres = 0 # 0.5 percentile
        elif sparsity_level in [2]:
            thres = 0.5244 # 0.70 percentile
        elif sparsity_level in [3]:
            thres = 0.8416 # 0.80 percentile
        else:
            assert False
        #
        per_zeros_bef = (np.sum(mat_scaled_sparse == 0) / np.prod(mat_scaled_sparse.shape)) * 100
        mat_scaled_sparse[mat_scaled < thres] = 0
        per_zeros_aft = (np.sum(mat_scaled_sparse == 0) / np.prod(mat_scaled_sparse.shape)) * 100
        #
        print("sparsity_level: ",sparsity_level,", thres: ",thres,", per_zeros_bef: ",per_zeros_bef,", per_zeros_aft: ",per_zeros_aft)
        #
        #make the matrix noisy
        #
        tot_num_entries = np.prod(mat_scaled_sparse.shape)
        if noise_level == 0:
            num_entries_to_flip = 0 # 0 percentage
        elif noise_level in [1]:
            num_entries_to_flip = int((0.05*tot_num_entries)/2) # 5%
        elif noise_level in [2]:
            num_entries_to_flip = int((0.10*tot_num_entries)/2) # 10%
        elif noise_level in [3]:
            num_entries_to_flip = int((0.20*tot_num_entries)/2) # 20%
        else:
            assert False
        #
        list_temp = []
        for temp_i in np.arange(mat_scaled_sparse.shape[0]):
            for temp_j in np.arange(mat_scaled_sparse.shape[1]):
                list_temp.append((temp_i,temp_j))
        #
        ij_sample = random.sample(list_temp, num_entries_to_flip)
        #
        rand_mu = self.__get_random_num2() #random mu, sigma chosen from a different range than the data generating distribution
        rand_sigma = self.__get_random_num2()
        for cur_ij in ij_sample:
            cur_i = cur_ij[0]
            cur_j = cur_ij[1]
            mat_scaled_sparse[cur_i,cur_j] += np.random.normal(rand_mu, rand_sigma)
        #
        print("freal: noise_level: ",noise_level,", tot_num_entries: ",tot_num_entries,", num_entries_to_flip: ",num_entries_to_flip)
        #
        return np.tanh(mat_scaled_sparse)
        #return self.__sigmoid(mat_scaled_sparse)
    #
    def __g_real(self, U_r,U_c,\
                sparsity_level, noise_level):
        #
        num_rows = U_r.shape[0]
        num_cols = U_c.shape[0]
        #
        mat_template = np.dot(U_r,U_c) 
        for i in np.arange(num_rows):
            for j in np.arange(num_cols):
                cur_row_entry = np.sum(U_r[i,:])
                cur_col_entry = np.sum(U_c[j,:])
                cur_mat_entry = cur_row_entry * cur_col_entry
                temp = cur_mat_entry**3 - 3*cur_mat_entry
                assert isinstance(temp, float)
                mat_template[i,j] = temp
        #
        print("#")
        print("mat_template.shape: ",mat_template.shape)
        print("#")
        return self.f(mat_template,\
                sparsity_level, noise_level)

    
    # def __g_real(self, U_r,U_c,\
    #             sparsity_level, noise_level):
    #     return self.f(np.dot(U_r,U_c),\
    #             sparsity_level, noise_level)
           
    #
    def fbin(self, x, sparsity_level, noise_level):
        x_non_linear_transf = (x**3 - 3*x) 
        scaler = StandardScaler()
        x_non_linear_transf_scaled = scaler.fit_transform(x_non_linear_transf)
        x_non_linear_transf_scaled_bin = x_non_linear_transf_scaled.copy()
        # make the matrix sparse
        if sparsity_level == 0:
            thres = -0.6745 # 0.25 percentile
        elif sparsity_level in [1]:
            thres = 0 # 0.5 percentile
        elif sparsity_level in [2]:
            thres = 0.5244 # 0.60 percentile
        elif sparsity_level in [3]:
            thres = 0.8416 # 0.80 percentile
        else:
            assert False
        #    
        per_zeros_bef = (np.sum(x_non_linear_transf_scaled_bin == 0) / np.prod(x_non_linear_transf_scaled_bin.shape)) * 100
        x_non_linear_transf_scaled_bin[x_non_linear_transf_scaled >= thres] = 1
        x_non_linear_transf_scaled_bin[x_non_linear_transf_scaled < thres] = 0
        per_zeros_aft = (np.sum(x_non_linear_transf_scaled_bin == 0) / np.prod(x_non_linear_transf_scaled_bin.shape)) * 100
        print("sparsity_level: ",sparsity_level,", thres: ",thres,", per_zeros_bef: ",per_zeros_bef,", per_zeros_aft: ",per_zeros_aft)
        #
        #make the matrix noisy
        #
        tot_num_entries = len(np.argwhere(x_non_linear_transf_scaled_bin == 1))
        if noise_level == 0:
            num_entries_to_flip = 0 # 0 percentage
        elif noise_level in [1]:
            num_entries_to_flip = int((0.05*tot_num_entries)/2) # 5%
        elif noise_level in [2]:
            num_entries_to_flip = int((0.10*tot_num_entries)/2) # 10%
        elif noise_level in [3]:
            num_entries_to_flip = int((0.20*tot_num_entries)/2) # 20%
        else:
            assert False
        #
        #pick random "num_entries_to_flip" number of 1s and change it to 0s
        # and viceversa
        #
        if not noise_level == 0:
            x_non_linear_transf_scaled_bin_noisy = self.__get_X_flipped(x_non_linear_transf_scaled_bin.copy(), num_entries_to_flip)
            assert not np.all(x_non_linear_transf_scaled_bin_noisy == x_non_linear_transf_scaled_bin)
            print("fbin: noise_level: ",noise_level,", tot_num_entries: ",tot_num_entries,", num_entries_to_flip: ",num_entries_to_flip)
        else:
            x_non_linear_transf_scaled_bin_noisy = x_non_linear_transf_scaled_bin
        
        #
        return x_non_linear_transf_scaled_bin_noisy
    #
    def __g_bin(self, U_r,U_c,\
                sparsity_level, noise_level):
        return self.fbin(np.dot(U_r,U_c),\
                    sparsity_level, noise_level)

    def get_data_dict(self):
        #generate a factor for each entity
        dict_entity_U = {}
        for cur_entity in self.dict_entity_size:
            dict_entity_U[cur_entity] = {}
            #
            cur_entity_size = self.dict_entity_size[cur_entity]
            cur_num_rows = self.dict_entity_size[cur_entity]
            cur_num_cols = self.k
            cur_mu = self.__get_random_num()
            cur_sigma = self.__get_random_num()
            #
            cur_U = self.__get_normal_matrix(cur_num_rows, cur_num_cols, cur_mu, cur_sigma)
            #cur_U = __get_random_matrix(cur_num_rows, cur_num_cols)
            dict_entity_U[cur_entity]["U"] = cur_U
            #
            print("cur_entity: ",cur_entity,", dict_entity_U[cur_entity].shape: ",cur_U.shape,\
                  ", min: ",np.round(np.min(cur_U),4),\
                  ", max: ",np.round(np.max(cur_U),4),\
                  ", mean: ",np.round(np.mean(cur_U),4),\
                  ", mu: ",np.round(cur_mu,4),\
                  ", sigma: ",np.round(cur_sigma,4))
            # 
            dict_entity_U[cur_entity]["mean"] = cur_mu
            dict_entity_U[cur_entity]["sigma"] = cur_sigma  
        #
        dict_mat = {}
        for cur_mat_id in self.G.keys():
            dict_mat[cur_mat_id] = {}
            #
            cur_row_entity = self.G[cur_mat_id][0]
            cur_col_entity = self.G[cur_mat_id][1]
            cur_mat_dtype = self.dict_mat_dtype[cur_mat_id]
            print("cur_mat: ",cur_mat_id,", cur_row_entity: ",cur_row_entity,", cur_col_entity: ",cur_col_entity,", cur_mat_dtype: ",cur_mat_dtype)
            #
            cur_row_U = dict_entity_U[cur_row_entity]["U"]
            cur_col_U = dict_entity_U[cur_col_entity]["U"]
            #
            if cur_mat_id in self.list_mod_mat_id:
                cur_sparsity_level = self.sparsity_level
                cur_noise_level = self.noise_level
            else:
                cur_sparsity_level = 0
                cur_noise_level = 0
            #
            if cur_mat_dtype in ["real"]:
                cur_mat = self.__g_real(cur_row_U, cur_col_U.T,\
                                       cur_sparsity_level, cur_noise_level)
            elif cur_mat_dtype in ["binary"]:
                cur_mat = self.__g_bin(cur_row_U, cur_col_U.T,\
                                       cur_sparsity_level, cur_noise_level)
            else:
                assert False
            #
            num_zeros = np.sum(np.sum(cur_mat == 0))
            per_zeros = np.round(num_zeros/np.prod(cur_mat.shape)*100,1)
            #
            print("cur_mat_id: ",cur_mat_id,", shape: ",cur_mat.shape,", min: ",np.round(np.min(cur_mat),4),\
                  ", max: ",np.round(np.max(cur_mat),4),\
                  ", mean: ",np.round(np.mean(cur_mat),4),\
                 ", %zeros: ",per_zeros,\
                 ", noise: ",cur_noise_level)
            print("#")
            #
            dict_mat[cur_mat_id]["X"] = cur_mat
            dict_mat[cur_mat_id]["per_zeros"] = per_zeros
            dict_mat[cur_mat_id]["noise_level"] = cur_noise_level
            dict_mat[cur_mat_id]["sparsity_level"] = cur_sparsity_level
            dict_mat[cur_mat_id]["min"] = np.round(np.min(cur_mat),4) 
            dict_mat[cur_mat_id]["max"] = np.round(np.max(cur_mat),4)
            dict_mat[cur_mat_id]["mean"] = np.round(np.mean(cur_mat),4)
            dict_mat[cur_mat_id]["dtype"] = cur_mat_dtype
            #
        return dict_mat