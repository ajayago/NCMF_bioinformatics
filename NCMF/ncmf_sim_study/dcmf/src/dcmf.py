import numpy as np
import pickle as pkl
import time
import itertools
import pprint
import scipy

from src.dcmf_base import dcmf_base
from src.base import base

import os

class dcmf(base):

    def __init__(self, G, X_data, X_meta, num_chunks,\
        k, kf, e_actf, d_actf,\
        learning_rate, weight_decay, convg_thres, max_epochs,\
        is_gpu=False, gpu_ids = "1",\
        is_pretrain=False, pretrain_thres=None, max_pretrain_epochs=None,\
        is_linear_last_enc_layer=False,is_linear_last_dec_layer=False,\
        X_val={}, val_metric="rmse",at_k=5, is_val_transpose=False,\
        num_val_sets=1):
        #flag that says the call is from dcmf and not dcmf_base
        base.__init__(self, G, X_data, X_meta, num_chunks,\
                            k, kf, e_actf, d_actf,\
                            learning_rate, weight_decay, convg_thres, max_epochs,\
                            is_gpu, gpu_ids,\
                            is_pretrain, pretrain_thres, max_pretrain_epochs,\
                            is_linear_last_enc_layer,is_linear_last_dec_layer,\
                            X_val, val_metric,at_k, is_val_transpose,num_folds=num_val_sets)
        self.is_dcmf_base = False
       	self.out_dict_info = {}
        self.out_dict_U = {}
        self.out_dict_X_prime = {}
        #
        self.is_bo = False #To perform validation accordingly
        self.validate_input()
        print("#")
        print("dCMF:")
        print("---")
        self.print_params()

    def fit(self):
        #
        loss_list_dict = {}
        val_perf_dict_of_dict = {}
        val_perf_total_dict = {}
        for fold_num in np.arange(1,self.num_folds+1):
            fold_num = str(fold_num)
            print("## fold_num: ",fold_num," ##")
            X_val_fold = {}
            for X_id in self.X_val.keys():
                #print("self.X_val[X_id].keys(): ",self.X_val[X_id].keys())
                X_val_fold[X_id] = self.X_val[X_id][fold_num]
            X_data_fold = {}                
            for X_id in self.X_data.keys():
                if X_id in self.X_val.keys():
                    X_data_fold[X_id] = self.X_data[X_id][fold_num]
                else:
                    X_data_fold[X_id] = self.X_data[X_id]
            #
            dcmf_model = dcmf_base(self.G, X_data_fold, self.X_meta, self.num_chunks,\
                                    self.k, self.kf, self.e_actf, self.d_actf,\
                                    self.learning_rate, self.weight_decay, self.convg_thres, self.max_epochs,\
                                    self.is_gpu, self.gpu_ids,\
                                    self.is_pretrain, self.pretrain_thres, self.max_pretrain_epochs,\
                                    self.is_linear_last_enc_layer,self.is_linear_last_dec_layer,\
                                    X_val_fold,self.val_metric,self.at_k,self.is_val_transpose)
            #
            dcmf_model.fit()
            loss_list_dict[fold_num] = dcmf_model.loss_list
            self.out_dict_U[fold_num] = dcmf_model.U_dict_
            self.out_dict_X_prime[fold_num] = dcmf_model.X_prime_dict_
            #
            val_perf_dict = dcmf_model.X_val_perf
            val_perf_dict_of_dict[fold_num] = val_perf_dict
            if self.val_metric in ["r@k","p@k"]:
                val_perf_total = 0
                for X_id in val_perf_dict.keys():
                    val_perf_total+=val_perf_dict[X_id][self.at_k]
            else:
                val_perf_total = np.sum(list(val_perf_dict.values()))
            val_perf_total_dict[fold_num] = val_perf_total
        #calc avg loss,perf
        avg_loss_list = np.mean(list(loss_list_dict.values()),axis=0)
        avg_val_perf_total = np.mean(list(val_perf_total_dict.values()))
        #
        avg_val_perf_dict = {}
        val_perf_dict_temp = {}
        for X_id in self.X_val.keys():
            val_perf_dict_temp[X_id] = []
        if self.val_metric in ["r@k","p@k"]:
            for fold_num in val_perf_dict_of_dict.keys():
                val_perf_dict_fold = val_perf_dict_of_dict[fold_num]
                for X_id in val_perf_dict_fold.keys():
                    val_perf_dict_temp[X_id].append(list(val_perf_dict_fold[X_id].values()))
            for X_id in val_perf_dict_temp.keys():
                avg_val_perf_temp = np.mean(val_perf_dict_temp[X_id],axis=0)
                avg_val_perf_at_k_temp = {}
                for temp_k in np.arange(1,self.at_k+1):
                    avg_val_perf_at_k_temp[temp_k] = avg_val_perf_temp[temp_k-1]
                avg_val_perf_dict[X_id] = avg_val_perf_at_k_temp
        else:
            for fold_num in val_perf_dict_of_dict.keys():
                val_perf_dict_fold = val_perf_dict_of_dict[fold_num]
                for X_id in val_perf_dict_fold.keys():
                    val_perf_dict_temp[X_id].append(val_perf_dict_fold[X_id])
            for X_id in val_perf_dict_temp.keys():
                avg_val_perf_temp = np.mean(val_perf_dict_temp[X_id],axis=0)
                avg_val_perf_dict[X_id] = avg_val_perf_temp
        #
        E = len(self.G.keys())
        M = len(self.X_data.keys())
        dict_params = {
                        "learning_rate":self.learning_rate,
                        "weight_decay":self.weight_decay,
                        "convg_thres":self.convg_thres,
                        "max_epochs":self.max_epochs,
                        "is_pretrain":self.is_pretrain,
                        "pretrain_thres":self.pretrain_thres,
                        "max_pretrain_epochs":self.max_pretrain_epochs,
                        "num_chunks":self.num_chunks,
                        "k":self.k,
                        "kf":self.kf,
                        "e_actf":self.e_actf,
                        "d_actf":self.d_actf,
                        "is_linear_last_enc_layer":self.is_linear_last_enc_layer,
                        "is_linear_last_dec_layer":self.is_linear_last_dec_layer
                    }
        self.out_dict_info = {"params":dict_params,
                        "num_val_sets":self.num_folds,
                        "loss_all_folds": loss_list_dict,
                        "loss_all_folds_avg_tuple":list(avg_loss_list),
                        "loss_all_folds_avg_sum":np.sum(avg_loss_list),
                        "val_metric":self.val_metric,
                        "val_perf_all_folds": val_perf_dict_of_dict,
                        "val_perf_all_folds_avg": avg_val_perf_dict,
                        "val_perf_all_folds_total":val_perf_total_dict,
                        "val_perf_all_folds_total_avg":avg_val_perf_total,
                        "E":E,
                        "M":M
                    }
        if self.val_metric in ["r@k","p@k"]:
            self.out_dict_info["at_k"] = self.at_k
        #

