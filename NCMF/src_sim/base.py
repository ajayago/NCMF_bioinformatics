import numpy as np
import itertools
import operator
import os
import pprint
import scipy

class base():

    def print_params(self):
        print("#")
        print("dCMF: ")
        print("#")
        print("learning_rate: ",self.learning_rate)
        print("weight_decay: ",self.weight_decay)
        print("convg_thres: ",self.convg_thres)
        print("max_epochs: ",self.max_epochs)
        print("isPretrain: ",self.is_pretrain)
        print("pretrain_thres: ",self.pretrain_thres)
        print("max_pretrain_epochs: ",self.max_pretrain_epochs)
        print("num_chunks: ",self.num_chunks)
        print("k: ",self.k)
        print("kf: ",self.kf)
        print("e_actf: ",self.e_actf)
        print("d_actf: ",self.d_actf)
        print("is_gpu: ",self.is_gpu)
        print("gpu_ids: ",self.gpu_ids)
        print("num entities: ", self.E)
        print("num matrices: ",self.M)
        print("num_val_sets: ",self.num_folds)
        print("X_val #matrices: ",len(self.X_val.keys()))
        print("val_metric (used only if X_val #matrices > 0): ",self.val_metric)
        print("at_k (used only if X_val #matrices > 0 and val_metric is r@k or p@k): ",self.at_k)
        print("is_val_transpose: ",self.is_val_transpose)
        print("is_linear_last_enc_layer: ",self.is_linear_last_enc_layer)
        print("is_linear_last_dec_layer: ",self.is_linear_last_dec_layer)
        print("#")

    def validate_input(self):
        #ensure if expected data types and structures are used
        assert isinstance(self.G, dict), "G must be a dictionary with key: entity ID, value: associated matrix ID"
        assert isinstance(self.X_data, dict), "X_data must be a dictionary with key: matrix ID, value: matrix"
        assert isinstance(self.X_meta, dict), "X_meta must be a dictionary with key: matrix ID, value: list with elements [row_entity_ID, column_entity_ID]"
        #G: ensure all IDs are strings, values are list of strings
        for e_id in self.G.keys():
            assert isinstance(e_id, str), "Entity IDs in G must be of type str. Got ("+str(type(e_id))+")"
        for e_id in self.G.keys():
            assert isinstance(self.G[e_id], list), "G["+str(e_id)+"] must be a list. Got ("+str(type(self.G[e_id]))+")"
            for temp_item in self.G[e_id]:
                assert isinstance(temp_item,str), "G["+str(e_id)+"] must be a list of str. Got ("+str(type(temp_item))+") in the list."
        #X_data: ensure all IDs are string, values are np.array
        for X_id in self.X_data.keys():
            assert isinstance(X_id, str), "Matrix IDs in X_data must be of type str. Got ("+str(type(X_id))+")"
            assert isinstance(self.X_data[X_id],dict) or isinstance(self.X_data[X_id],np.ndarray),"X_data must be either np.ndarray or (if this matrix participates in validation, then) a dict(1) of dict(2), where dict(1)'s keys are matrix id and values are dict(2). The dict(2)'s keys are fold id and values are nd-array"
        #X_meta: ensure all IDs are string, values are list of 2 IDs
        for X_id in self.X_meta.keys():
            assert isinstance(X_id, str), "Matrix IDs in X_meta must be of type str. Got ("+str(type(X_id))+")"
            assert isinstance(self.X_meta[X_id],list), "Values in X_meta must be of type list. Got ("+str(type(self.X_meta[X_id]))+") for matrix ID: "+X_id
            assert len(self.X_meta[X_id]) == 2, "Values in X_meta must be of type list and size 2. Got a list of length ("+str(len(self.X_meta[X_id]))+") for matrix ID: "+X_id
            for temp_item in self.X_meta[X_id]:
                assert isinstance(temp_item, str), "Values in X_meta must be list of str. Got a list with ("+str(type(temp_item))+") item for matrix ID: "+X_id
        #ensure e and X labels used across the data structures are consistent and tally
        G_e_list = list(np.unique(list(self.G.keys())))
        G_x_list = list(np.unique(list(itertools.chain(*list(self.G.values())))))
        X_meta_e_list = list(np.unique(list(itertools.chain(*list(self.X_meta.values())))))
        X_meta_x_list = list(np.unique(list(self.X_meta.keys())))
        X_data_x_list = list(np.unique(list(self.X_data.keys())))
        if G_e_list != X_meta_e_list:
            print("e in G: ",G_e_list)
            print("e in X_meta",X_meta_e_list)
            raise Exception("Entity IDs in G and X_meta should match")   
        if G_x_list != X_meta_x_list or \
            X_meta_x_list != X_data_x_list:
            print("X in G: ",G_x_list)
            print("X in X_meta: ",X_meta_x_list)
            print("X in X_data: ",X_data_x_list)
            raise Exception("Matrix IDs in G, X_data and X_meta should match")
        #val matrices
        if len(self.X_val.keys()) > 0:
            assert (self.num_folds > 0) and isinstance(self.num_folds,int) and self.num_folds is not None, "num_val_sets must be an int > 0. Got: "+str(self.num_folds)
            if self.num_folds == 1 and self.is_dcmf_base:        
                assert (self.val_metric in ["rmse","r@k","p@k","auc"]),"val_metric can only be one of the following values: rmse,r@k,p@k,auc"
                #
                unmatched_X_id_set = set(self.X_val.keys())-set(self.X_data.keys())
                assert len(unmatched_X_id_set) == 0,"The following matrix IDs in x_val are not present in the input X_data. Missing IDs: "+str(unmatched_X_id_set)+", X_data IDs: "+str(set(self.X_data.keys()))
                if self.val_metric == "auc":
                    temp_err_msg = "If val_metric is 'auc', then X_val values should be list of triplets (i,j,x), where i - row idx, j - col idx, x - real cell value. Both i,j must be int starting with 0. Got: "
                    for X_id in self.X_val.keys():
                        assert isinstance(self.X_val[X_id],list),temp_err_msg+str(type(self.X_val[X_id]))+" for X_id: "+str(X_id)
                        for temp_triplet in self.X_val[X_id]:
                            assert len(temp_triplet) == 3,temp_err_msg+str(temp_triplet)+" for X_id: "+str(X_id)
                            assert isinstance(temp_triplet[0],int),temp_err_msg+str(temp_triplet)+" for X_id: "+str(X_id)
                            assert isinstance(temp_triplet[1],int),temp_err_msg+str(temp_triplet)+" for X_id: "+str(X_id)
                            assert np.isreal(temp_triplet[2]),temp_err_msg+str(temp_triplet)+" for X_id: "+str(X_id)
                else:
                    for X_id in self.X_val.keys():
                        assert scipy.sparse.issparse(self.X_val[X_id]),"For val_metric: "+str(self.val_metric)+" X_val should be a scipy sparse matrix."
                if self.val_metric in ["r@k","p@k"]:
                    assert (self.at_k is not None) and isinstance(self.at_k,int) and self.at_k > 0,"If val_metric is one of ['r@k','p@k'] then a positive int at_k must be provided. Got: "+str(self.at_k)
                assert isinstance(self.is_val_transpose,bool),"is_val_transpose can only either be True or False"
            else:
                #check if for all x_data matrices with val sets i.e. provided as dict, there are corresponding x_val matrices
                count_dicts = 0
                temp_list_x_id_with_dict_values = []
                for X_id in self.X_data.keys():
                    if isinstance(self.X_data[X_id],dict):
                        count_dicts+=1
                        temp_list_x_id_with_dict_values.append(X_id)
                if count_dicts > len(self.X_val.keys()):
                    assert False, "The validation sets for "+str(count_dicts-len(self.X_val.keys()))+" X_data matrices with ID: "+str(set(temp_list_x_id_with_dict_values)-set(self.X_val.keys()))+" are missing in X_val."
                elif count_dicts < len(self.X_val.keys()):
                    assert False, "The validation sets for "+str(len(self.X_val.keys())-count_dicts)+" X_val matrices with ID: "+str(set(self.X_val.keys())-set(temp_list_x_id_with_dict_values))+" are missing in X_data."    
                for X_val_id in self.X_val.keys():
                    assert isinstance(self.X_val[X_val_id],dict),"X_val must be a dict(1) of dict(2), where dict(1)'s keys are matrix id and values are dict(2). The dict(2)'s keys are fold id and values are nd-array"
                    assert len(self.X_val[X_val_id].keys()) == self.num_folds,"The num_val_sets did not match input X_val data folds for id: "+str(X_val_id)+". len(X_val[X_val_id].keys()): "+str(len(self.X_val[X_val_id].keys()))+", num_val_sets: "+str(self.num_folds)
                    assert isinstance(self.X_data[X_val_id],dict),"X_data for id: "+str(X_val_id)+", must be a dict(1) of dict(2), where dict(1)'s keys are matrix id and values are dict(2). The dict(2)'s keys are fold id and values are nd-array. Got: "+str(type(self.X_data[X_val_id]))
                    assert len(self.X_data[X_val_id].keys()) == self.num_folds,"The num_val_sets did not match input X_data folds for id: "+str(X_val_id)+". len(X_data[X_val_id].keys()): "+str(len(self.X_data[X_val_id].keys()))+", num_val_sets: "+str(self.num_folds)
                    for fold_num in self.X_data[X_val_id].keys():
                        assert isinstance(self.X_data[X_val_id][fold_num], np.ndarray), "Matrix for fold_num: "+str(fold_num)+" and id: "+str(X_val_id)+" is not ndarray. Got ("+str(type(self.X_data[X_val_id][fold_num]))+")"
                #
                assert (self.val_metric in ["rmse","r@k","p@k","auc"]),"val_metric can only be one of the following values: rmse,r@k,p@k,auc"
                #
                unmatched_X_id_set = set(self.X_val.keys())-set(self.X_data.keys())
                assert len(unmatched_X_id_set) == 0,"The following matrix IDs in x_val are not present in the input X_data. Missing IDs: "+str(unmatched_X_id_set)+", X_data IDs: "+str(set(self.X_data.keys()))
                if self.val_metric == "auc":
                    temp_err_msg = "If val_metric is 'auc', then X_val values should be list of triplets (i,j,x), where i - row idx, j - col idx, x - real cell value. Both i,j must be int starting with 0. Got: "
                    for X_id in self.X_val.keys():
                        for fold_num in self.X_val[X_id].keys():
                            assert isinstance(self.X_val[X_id][fold_num],list),temp_err_msg+str(type(self.X_val[X_id][fold_num]))+" for X_id: "+str(X_id)+" for validation set with id: "+str(fold_num)
                            list_triplets = self.X_val[X_id][fold_num]
                            for temp_triplet in list_triplets:
                                assert len(temp_triplet) == 3,temp_err_msg+str(temp_triplet)+" for X_id: "+str(X_id)+" and fold: "+str(fold_num)
                                assert isinstance(temp_triplet[0],int),temp_err_msg+str(temp_triplet)+" for X_id: "+str(X_id)+" and fold: "+str(fold_num)
                                assert isinstance(temp_triplet[1],int),temp_err_msg+str(temp_triplet)+" for X_id: "+str(X_id)+" and fold: "+str(fold_num)
                                assert np.isreal(temp_triplet[2]),temp_err_msg+str(temp_triplet)+" for X_id: "+str(X_id)+" and fold: "+str(fold_num)
        else:
            print("WARNING: The following parameters are unused since no validation data provided.")
            print("val_metric: ",self.val_metric)
            print("at_k: ",self.at_k)
            print("is_val_transpose: ",self.is_val_transpose)

        #these can't be none if is_bo == False
        list_of_mandatory_params = ["num_chunks","k","kf","e_actf","d_actf","learning_rate","weight_decay","convg_thres", "max_epochs"] 
        for param_name in list_of_mandatory_params:
            f = operator.attrgetter(param_name)
            if not self.is_bo and f(self) is None:
                assert False, "param: "+param_name+" can't be None."
        #pretrain
        assert isinstance(self.is_pretrain, bool), "is_pretrain can either be True or False"
        if self.is_pretrain:
            assert (self.pretrain_thres is not None) and (self.max_pretrain_epochs is not None)," If is_pretrain == True, then pretrain_thres and max_pretrain_epochs should not be None."
            assert np.isreal(self.pretrain_thres),"pretrain_thres must be real"
            assert isinstance(self.max_pretrain_epochs, int) or (self.max_pretrain_epochs == None), "max_pretrain_epochs can be either None(to run till convergence) or an int"
        #
        assert np.isreal(self.kf)
        assert self.kf > 0 and self.kf < 1, "kf must be in range (0,1)"
        assert isinstance(self.k, int), "k (the encoding length) must be an int"
        assert isinstance(self.num_chunks, int), "num_chunks must be an int"
        assert self.num_chunks >= 1, "num_chunks must be an int and >= 1"
        assert isinstance(self.e_actf, str), "e_actf must be a str"
        assert isinstance(self.d_actf, str), "d_actf must be a str"
        assert np.isreal(self.learning_rate), "learning_rate must be real"
        assert np.isreal(self.weight_decay), "weight_decay must be real"
        assert np.isreal(self.convg_thres), "convg_thres must be real"
        assert isinstance(self.max_epochs, int) or (self.max_epochs == None), "max_epochs can be either None(to run till convergence) or an int"
        assert isinstance(self.is_gpu, bool), "is_gpu can either be True or False"
        assert isinstance(self.gpu_ids, str), "gpu_ids has to be a str like '1' or '1,2' where 1 and 2 are the gpu cuda IDs" 
        assert isinstance(self.is_linear_last_enc_layer, bool), "is_linear_last_enc_layer can either be True or False"
        assert isinstance(self.is_linear_last_dec_layer, bool), "is_linear_last_dec_layer can either be True or False"

    def __init__(self, G, X_data, X_meta, num_chunks,\
        k, kf, e_actf, d_actf,\
        learning_rate, weight_decay, convg_thres, max_epochs,\
        is_gpu, gpu_ids,\
        is_pretrain, pretrain_thres, max_pretrain_epochs,\
        is_linear_last_enc_layer,is_linear_last_dec_layer,\
        X_val, val_metric,at_k, is_val_transpose,num_folds):
        print("dcmf_base.__init__ - start")
        #outputs
        self.U_dict_ = {}
        self.X_prime_dict_ = {}
        #inputs
        self.G = G
        self.X_data = X_data
        self.X_meta = X_meta
        self.X_val = X_val
        #hyperparams
        #learning algo
        self.learning_rate = learning_rate
        self.weight_decay=weight_decay
        self.convg_thres = convg_thres
        self.max_epochs = max_epochs
        self.is_pretrain = is_pretrain
        self.pretrain_thres = pretrain_thres
        self.max_pretrain_epochs = max_pretrain_epochs
        #data
        self.num_chunks = num_chunks
        #network
        self.k = k
        self.kf = kf
        self.e_actf = e_actf
        self.d_actf = d_actf
        self.is_linear_last_enc_layer = is_linear_last_enc_layer
        self.is_linear_last_dec_layer = is_linear_last_dec_layer
        self.E = len(G.keys())
        self.M = len(X_data.keys())
        #bookkeeping
        self.dict_epoch_loss = {}
        self.dict_epoch_aec_rec_loss = {}
        self.dict_epoch_mat_rec_loss = {}
        #gpu
        self.is_gpu = is_gpu
        self.gpu_ids = gpu_ids
        #val
        self.val_metric = val_metric
        self.at_k = at_k
        self.is_val_transpose = is_val_transpose
        #check type and format 
        self.is_bo = False #To perform validation accordingly
        self.is_dcmf_base = False
        self.num_folds = num_folds
        #set the gpu_id to use
        if self.is_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"]=self.gpu_ids
        #concatenated-matrix
        self.C_dict = {}
        self.C_dict_chunks = {}
        #pytorch variable version of the input matrices
        self.X_data_var = {}
        #Network
        self.N_aec_dict = {}
        #loss
        self.loss_list = [] #list of losses, loss/elements of the list corresponds to tasks
        self.dict_epoch_loss = {}
        self.dict_epoch_aec_rec_loss = {}
        self.dict_epoch_mat_rec_loss = {}
        #pretrain loss 
        self.pretrain_dict_epoch_loss = {}
        self.pretrain_dict_epoch_aec_rec_loss = {}
        # validation set performance
        self.X_val_perf = {}
        self.pp = pprint.PrettyPrinter()
        print("dcmf_base.__init__ - end")
