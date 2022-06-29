from src.base import base
from src.dcmf import dcmf
from src import bo_config

from src.hyperopt import mtgp
from src.hyperopt import objective_multi_output
from src.hyperopt import AcquisitionEI_multi_output

import itertools
import operator
import os
import numpy as np
import scipy

import GPy
import GPyOpt

class dcmf_bo(base):

    def __print_params(self):
        list_bo_hyperparams = list(self.bo_hyperparams_dict.keys())
        print("Input Hyperparameters:-")
        if "learning_rate" not in list_bo_hyperparams:
            print("learning_rate: ",self.learning_rate)
        if "weight_decay" not in list_bo_hyperparams:
            print("weight_decay: ",self.weight_decay)
        if "convg_thres" not in list_bo_hyperparams:
            print("convg_thres: ",self.convg_thres)
        if "max_epochs" not in list_bo_hyperparams:
            print("max_epochs: ",self.max_epochs)
        if "is_pretrain" not in list_bo_hyperparams:
            print("isPretrain: ",self.is_pretrain)
        if "pretrain_thres" not in list_bo_hyperparams:
            print("pretrain_thres: ",self.pretrain_thres)
        if "max_pretrain_epochs" not in list_bo_hyperparams:
            print("max_pretrain_epochs: ",self.max_pretrain_epochs)
        if "num_chunks" not in list_bo_hyperparams:
            print("num_chunks: ",self.num_chunks)
        if "k" not in list_bo_hyperparams:
            print("k: ",self.k)
        if "kf" not in list_bo_hyperparams:
            print("kf: ",self.kf)
        if "e_actf" not in list_bo_hyperparams:
            print("d_actf: ",self.e_actf)
        if "d_actf" not in list_bo_hyperparams:
            print("d_actf: ",self.d_actf)                     
        print("---")
        print("Hyperparameters to be set using BO:-")
        for temp_hyp in list_bo_hyperparams:
            print(temp_hyp)
        print("---")
        print("val:-")
        print("num_val_sets: ",self.num_folds)
        print("X_val #matrices: ",len(self.X_val.keys()))
        print("best_criterion: ",self.best_criterion)
        print("val_metric (used only if X_val #matrices > 0): ",self.val_metric)
        print("at_k (used only if X_val #matrices > 0 and val_metric is r@k or p@k): ",self.at_k)
        print("is_val_transpose: ",self.is_val_transpose)   
        print("---")
        print("Others:-")
        print("is_gpu: ",self.is_gpu)
        print("gpu_ids: ",self.gpu_ids)
        print("num entities: ", self.E)
        print("num matrices: ",self.M)
        print("is_linear_last_enc_layer: ",self.is_linear_last_enc_layer)
        print("is_linear_last_dec_layer: ",self.is_linear_last_dec_layer)
        print("---")
 
    def __validate_input_bo(self):
        if len(self.X_val.keys()) > 0:
            assert (self.best_criterion in ["loss","val"]),"best_criterion can only either be 'loss' or 'val'"
        else:
            assert (self.best_criterion == "loss"),"best_criterion can only be 'loss' if no validation data is provided"
        #
        self.validate_input()
        #
        list_possible_bo_params = ["learning_rate","convg_thres","weight_decay","kf","k","num_chunks","pretrain_thres"]
        for param_name in list_possible_bo_params:
            f = operator.attrgetter(param_name)
            if f(self) is None:
                self.bo_hyperparams_dict[param_name] = None
        #
        list_of_mandatory_params = ["e_actf","d_actf","max_epochs"] 
        for param_name in list_of_mandatory_params:
            f = operator.attrgetter(param_name)
            if f(self) is None:
                assert False, "param: "+param_name+" can't be None."
        #Check if the config is available for all the hyperparameters to be selected using BO
        list_bo_hyperparams = list(self.bo_hyperparams_dict.keys())
        assert len(list_bo_hyperparams) > 0,"No hyperparameters selected for search using BO. Please (1) use 'dcmf' API if you intent to provide all hyperparameters manually,"+\
                                            " or (2) select hyperparamters to searcch from the following list and set them None while initiating dcmf_bo.  Hyperparameters list: "+str(list_possible_bo_params)         
        bo_config_bounds = bo_config.bounds
        list_hyp_config = []
        for temp_obj in bo_config_bounds:
            list_hyp_config.append(temp_obj["name"]) #sample temp_obj: {"name": "learning_rate", "type": "continuous", "domain": (1e-6,1e-4)}
        missing_hyp_config = set(list_bo_hyperparams) - set(list_hyp_config)
        assert len(missing_hyp_config) == 0, "GPyOpt config missing for the following hyperparameters: \n "+str(missing_hyp_config)
   
    def __init__(self, G, X_data, X_meta, num_chunks,\
        k, kf, e_actf, d_actf,\
        learning_rate, weight_decay, convg_thres, max_epochs,\
        is_gpu=False, gpu_ids = "1",\
        is_pretrain=False, pretrain_thres=None, max_pretrain_epochs=None,\
        is_linear_last_enc_layer=False,is_linear_last_dec_layer=False,\
        X_val={}, val_metric="rmse",at_k=5, is_val_transpose=False,\
        best_criterion="loss",num_bo_steps=2, initial_design_size=2,num_val_sets=1):
        #
        base.__init__(self, G, X_data, X_meta, num_chunks,\
                            k, kf, e_actf, d_actf,\
                            learning_rate, weight_decay, convg_thres, max_epochs,\
                            is_gpu, gpu_ids,\
                            is_pretrain, pretrain_thres, max_pretrain_epochs,\
                            is_linear_last_enc_layer,is_linear_last_dec_layer,\
                            X_val, val_metric,at_k, is_val_transpose,num_folds=num_val_sets)
        self.is_bo = True #To perform validation accordingly
        #
        print("dcmf_bo.__init__ - start")
        #outputs
        self.out_dict_p_hash_info = {}
        self.out_list_D = []
        #val
        self.best_criterion = best_criterion
        #BO
        self.bo_hyperparams_dict = {}
        self.num_bo_steps = num_bo_steps
        self.initial_design_size = initial_design_size
        #check type and format 
        self.__validate_input_bo()
        #set the gpu_id to use
        if self.is_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"]=self.gpu_ids
        #debug
        print("dCMF + BO:")
        print("---")
        self.__print_params()
        print("dcmf_bo.__init__ - end")

    def __get_best_params(self,X,Y,rmse_dict_list):
        temp_rmse_list = []
        for i in np.arange(len(rmse_dict_list)):
            if self.best_criterion == "loss":
                temp_rmse_list.append(rmse_dict_list[i]["loss_all_folds_avg_sum"])
            elif self.best_criterion == "val":
                temp_rmse_list.append(rmse_dict_list[i]["val_perf_all_folds_total_avg"])
            else:
                assert False, "Unknown best parameter selection criterion:"+str(self.best_criterion)
        if self.best_criterion == "loss":
            best_idx = np.argmin(np.array(temp_rmse_list))
        else:
            #"val"
            if self.val_metric in ["rmse"]:
                best_idx = np.argmin(np.array(temp_rmse_list))
            else:     
                #auc,r@k,p@k
                best_idx = np.argmax(np.array(temp_rmse_list))
        best_X = X[best_idx]
        best_Y = Y[best_idx]
        best_rmse_dict = rmse_dict_list[best_idx]
        best_rmse_dict["best_criterion"] = self.best_criterion
        return best_X,best_Y,best_rmse_dict

    def run_dcmf(self,x):
        list_bo_hyperparams = list(self.bo_hyperparams_dict.keys())
        params_list_config_order = ["learning_rate","convg_thres","weight_decay","kf","k","num_chunks","pretrain_thres"]
        x_params_list = x[0]
        x_params_dict = {}
        j = 0
        for i in np.arange(len(x_params_list)):
            if params_list_config_order[i] in list_bo_hyperparams:
                x_params_dict[params_list_config_order[i]] = x_params_list[j]
                j+=1
        #
        if "learning_rate" in list_bo_hyperparams:
            self.learning_rate = x_params_dict["learning_rate"]
        #
        if "weight_decay" in list_bo_hyperparams:
            self.weight_decay = x_params_dict["weight_decay"]
        #
        if "convg_thres" in list_bo_hyperparams:
            self.convg_thres = x_params_dict["convg_thres"]
        #
        if "pretrain_thres" in list_bo_hyperparams:
            self.pretrain_thres = x_params_dict["pretrain_thres"]
        #
        if "num_chunks" in list_bo_hyperparams:
            self.num_chunks = x_params_dict["num_chunks"]
        #
        if "k" in list_bo_hyperparams:
            self.k = x_params_dict["k"]
        #
        if "kf" in list_bo_hyperparams:
            self.kf = x_params_dict["kf"]
        #
        dcmf_model = dcmf(self.G, self.X_data, self.X_meta, self.num_chunks,\
                                self.k, self.kf, self.e_actf, self.d_actf,\
                                self.learning_rate, self.weight_decay, self.convg_thres, self.max_epochs,\
                                self.is_gpu, self.gpu_ids,\
                                self.is_pretrain, self.pretrain_thres, self.max_pretrain_epochs,\
                                self.is_linear_last_enc_layer,self.is_linear_last_dec_layer,\
                                self.X_val,self.val_metric,self.at_k,self.is_val_transpose,self.num_folds)
      
        dcmf_model.fit()
        dcmf_model.out_dict_info["list_bo_hyperparams"] = list_bo_hyperparams
        return np.atleast_2d(dcmf_model.out_dict_info["loss_all_folds_avg_tuple"]),dcmf_model.out_dict_info

    def fit(self):
        print("dcmf_bo.fit - start")
        objective = objective_multi_output.SingleObjectiveMultiOutput(self.run_dcmf)
        surrogate_model = mtgp.MTGPModel(self.num_bo_steps)
        space = GPyOpt.Design_space(space=bo_config.bounds)
        aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
        initial_design = GPyOpt.experiment_design.initial_design('random', space, self.initial_design_size)
        acquisition = AcquisitionEI_multi_output.AcquisitionEIMO(surrogate_model, space, optimizer=aquisition_optimizer)
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
        bo = GPyOpt.methods.ModularBayesianOptimization(surrogate_model, space, objective, acquisition, evaluator, initial_design, normalize_Y=False)
        bo.run_optimization(max_iter=self.num_bo_steps, verbosity=True,eps = 0)
        #
        x_best,y_best,best_params_dict = self.__get_best_params(bo.X,bo.Y,bo.rmse_dict_list)
        #
        self.out_dict_p_hash_info = best_params_dict
        self.out_list_D = bo.rmse_dict_list
        print("dcmf_bo.fit - end")

