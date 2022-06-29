import numpy as np
import pickle as pkl
import time
import itertools
import pprint
import scipy

import torch
from torch.autograd import Variable

from src.aec import autoencoder
from src.base import base

from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

class dcmf_base(base):

    def __input_transformation(self):
        #for each entity, construct a concatenated matrix (as input to the corresponding autoencoder)
        #Building C_dict
        print("__input_transformation - start")
        print("#")
        print("concatenated-matrix construction...")
        for e_id in self.G.keys():
            print("e_id: ",e_id)
            X_id_list = self.G[e_id]
            print("X_id_list: ",X_id_list)
            X_data_list = []
            for X_id in X_id_list:
                print("X_id: ",X_id)
                print("X[X_id].shape: ",self.X_data[X_id].shape)
                if self.X_meta[X_id][0] == e_id:
                    X_data_list.append(self.X_data[X_id])
                else:
                    X_data_list.append(self.X_data[X_id].T)
            C_temp = Variable(torch.from_numpy(np.concatenate(X_data_list,axis=1)).float(),requires_grad=False)
            if self.is_gpu:
                self.C_dict[e_id] = C_temp.cuda()
            else:
                self.C_dict[e_id] = C_temp
            print("C_dict[e].shape: ",self.C_dict[e_id].shape)
            print("---")
        print("#")
        print("concatenated-matrix chunking...")
        #chunking
        #assert that num_chunks is not > number of datapoints
        min_num_datapoints = self.C_dict[e_id].shape[0] #using the latest e_id to init
        min_e_id_num_chunks = e_id
        #warn if the encoding length k is not > than minimum of the feature lengths
        min_features = self.C_dict[e_id].shape[0] #using the latest e_id to init
        min_e_id_k = e_id
        for e_id in self.G.keys():
            #num_chunks
            temp_num = self.C_dict[e_id].shape[0]
            if min_num_datapoints > temp_num:
                min_num_datapoints = temp_num
                min_e_id_num_chunks = e_id
            #k
            temp_feat = self.C_dict[e_id].shape[1]
            if min_features > temp_feat:
                min_features = temp_feat
                min_e_id_k = e_id
        assert (self.num_chunks <= min_num_datapoints), \
                "The num_chunks must be <= minimum entity size in the setting. Entity with ID: "+str(min_e_id_num_chunks)+\
                " is of minimum size "+str(min_num_datapoints)+". The num_chunks "+str(self.num_chunks)+" is larger than minimim entity size."
        if (self.k >= min_features):
                print("WARNING: Entity with ID: "+str(min_e_id_k)+\
                " has minimum feature size "+str(min_features)+" in the setting. The encoding length k "+str(self.k)+" is larger than minimim entity feature size.")
        print("#")
        print("e_id: ",min_e_id_num_chunks,", min_num_datapoints: ",min_num_datapoints,", num_chunks: ",self.num_chunks)
        print("e_id: ",min_e_id_k,", min_features: ",min_features,", k: ",self.k)
        print("#")
        #Building C_dict_chunks 
        for e_id in self.C_dict.keys():
            print("e_id: ",e_id," C_dict[e_id].shape: ",self.C_dict[e_id].shape)
            C_temp = self.C_dict[e_id]
            C_temp_chunks_list = torch.chunk(C_temp,self.num_chunks,dim=0)
            print("C_temp_chunks_list[0].shape: ",C_temp_chunks_list[0].shape)
            self.C_dict_chunks[e_id] = C_temp_chunks_list
            print("---")      
        print("#")
        print("creating pytorch variables of input matrices...")
        #Convert input matrices to pytorch variables (to calculate reconstruction loss)
        #Building X_data_var
        for X_id in self.X_data.keys():
            X_temp = Variable(torch.from_numpy(self.X_data[X_id]).float(),requires_grad=False)
            if self.is_gpu:
                self.X_data_var[X_id] = X_temp.cuda()
            else:
                self.X_data_var[X_id] = X_temp
        print("#")
        print("__input_transformation - end")

    def __get_k_list(self,n,k,kf):
        k_list = []
        while True:
            k1 = int(n * kf)
            if k1 > k:
                k_list.append(k1)
                n = k1
            else:
                k_list.append(k)
                break
        return k_list

    def __get_actf_list(self,k_list,e_actf,d_actf):
        actf_list_e = []
        actf_list_d = []
        for k in k_list:
            actf_list_e.append(e_actf) 
            actf_list_d.append(d_actf)
        actf_list = actf_list_e+actf_list_d 
        return actf_list

    def __is_converged(self,prev_cost,cost,convg_thres):
        diff = (prev_cost - cost)
        if (abs(diff)) < convg_thres:
            return True

    def __network_construction(self):
        #for each entity construct an autoencoder
        #Building aec_dict
        print("__network_construction - start") 
        for e_id in self.G.keys():
            #print("e_id: ",e_id)
            C = self.C_dict[e_id]
            k_list = self.__get_k_list(C.shape[1],self.k,self.kf) 
            actf_list = self.__get_actf_list(k_list,self.e_actf,self.d_actf)
            aec = autoencoder(C.shape[1],k_list,actf_list,\
                                self.is_linear_last_enc_layer,self.is_linear_last_dec_layer)
            #print("aec: ")
            #print(aec)
            if self.is_gpu:
                aec.cuda()
            self.N_aec_dict[e_id] = aec
            #print("#")
        print("__network_construction - end")

    def __init__(self, G, X_data, X_meta, num_chunks,\
        k, kf, e_actf, d_actf,\
        learning_rate, weight_decay, convg_thres, max_epochs,\
        is_gpu=False, gpu_ids = "1",\
        is_pretrain=False, pretrain_thres=None, max_pretrain_epochs=None,\
        is_linear_last_enc_layer=False,is_linear_last_dec_layer=False,\
        X_val={}, val_metric="rmse",at_k=10, is_val_transpose=False):
        
        base.__init__(self, G, X_data, X_meta, num_chunks,\
                                k, kf, e_actf, d_actf,\
                                learning_rate, weight_decay, convg_thres, max_epochs,\
                                is_gpu, gpu_ids,\
                                is_pretrain, pretrain_thres, max_pretrain_epochs,\
                                is_linear_last_enc_layer,is_linear_last_dec_layer,\
                                X_val, val_metric,at_k, is_val_transpose,num_folds=1)
        #flag that says the call is from dcmf or dcmf_base
        self.is_dcmf_base = True
        self.validate_input()
        self.print_params()

    def __pretrain(self):
        #setup pretrain opt
        print("__pretrain - start")
        criterion = torch.nn.MSELoss()
        model_params = []
        for e_id in self.G.keys():
            params_temp = list(self.N_aec_dict[e_id].parameters())
            #print("aec for e_id: ",e_id,", #params: ",len(params_temp))
            model_params+=params_temp
        optimizer = torch.optim.Adam(model_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        #main loop - training - start
        self.pretrain_dict_epoch_loss = {}
        self.pretrain_dict_epoch_aec_rec_loss = {}
        epoch = 1
        prev_loss_epoch = 0
        while True:
            if epoch > self.max_pretrain_epochs:
                break
            #epoch - start
            s = time.time()
            #init - chunks to epoch loss
            loss_epoch = 0
            loss_aec_rec_epoch = {}
            for e_id in self.G.keys():
                loss_aec_rec_epoch[e_id] = 0
            #chunks processing - start
            for i in np.arange(self.num_chunks):
                #load current batch data 
                C_chunk_batch = {}
                for e_id in self.G.keys():
                    C_chunk_batch[e_id] = self.C_dict_chunks[e_id][i] #load current batch input
                #train
                #autoencoder reconstruction - chunk
                C_chunk_rec_dict = {}
                U_chunk_dict = {} #unused
                for e_id in self.G.keys():
                    C_chunk_rec_dict[e_id],U_chunk_dict[e_id] = self.N_aec_dict[e_id](C_chunk_batch[e_id])
                #loss
                #autoencoder reconstruction loss - chunk
                loss_aec_rec_dict = {}
                for e_id in C_chunk_batch.keys():
                    loss_aec_rec_dict[e_id] = criterion(C_chunk_rec_dict[e_id],C_chunk_batch[e_id])
                #sum all losses
                loss = np.sum(list(loss_aec_rec_dict.values()))
                #backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                #update - loss info
                loss_epoch += loss.item()
                for e_id in self.G.keys():
                    loss_aec_rec_epoch[e_id] += loss_aec_rec_dict[e_id].item()
            #chunks processing - end
            self.pretrain_dict_epoch_loss[epoch] = loss_epoch
            self.pretrain_dict_epoch_aec_rec_loss[epoch] = loss_aec_rec_epoch
            #epoch - end
            e = time.time()
            print("pretrain epoch: ",epoch," total loss L: ",loss_epoch," Took ",round(e-s,1)," secs.")
            #update - counter
            epoch+=1
            if self.__is_converged(prev_loss_epoch,loss_epoch,self.pretrain_thres):
                print("**pretrain converged**")
                break
            prev_loss_epoch = loss_epoch            
        #main loop - training - end
        print("__pretrain - end")

    def __get_auc(self,Rpred,Rtest_triplets):
        if self.is_val_transpose:
            Rpred = Rpred.T
        print("Computing AUC.")
        print("Rpred.shape: ",Rpred.shape)
        print("Rtest_triplets.shape: ",np.array(Rtest_triplets).shape)
        list_pred = []
        list_test = []
        for temp_triplet in Rtest_triplets:
            list_test.append(temp_triplet[2])
            list_pred.append(Rpred[temp_triplet[0],temp_triplet[1]])
        return roc_auc_score(list_test, list_pred)

    def __get_rmse(self,Rpred,Rtest):
        if self.is_val_transpose:
            Rpred = Rpred.T
        print("Computing RMSE.")
        print("Rpred.shape: ",Rpred.shape)
        print("Rtest.shape: ",Rtest.shape)
        row_idx_list,col_idx_list = Rtest.nonzero()
        nz_list_pred = []
        nz_list_test = []
        for i in np.arange(len(row_idx_list)):
            r = row_idx_list[i]
            c = col_idx_list[i]
            nz_list_pred.append(Rpred[r,c])
            nz_list_test.append(Rtest[r,c])
        print("nz_list_pred.shape: ",len(nz_list_pred))
        print("nz_list_test.shape: ",len(nz_list_test))
        rmse = np.sqrt(mean_squared_error(nz_list_test,nz_list_pred))
        ##mse = np.sum(np.absolute(nz_list_test-nz_list_pred))
        print("rmse: ",rmse)
        print("---")
        return rmse

    def __get_prob_at_k(self,Rpred,Rtest,at_k):
        if self.is_val_transpose:
            Rpred = Rpred.T
        print("get_prob_at_"+str(at_k)+": ")
        print("WARNING: For gene-disease prioritization ensure that the inpur Rpred and Rtest are with row entity: disease and col_entity: genes")
        print("Rpred.shape: ",Rpred.shape)
        print("Rtest.shape: ",Rtest.shape)
        test_rank_list = []
        for ij in np.argwhere(Rtest):
            i = ij[0]
            j = ij[1]
            cur_row_rank = (-Rpred[i]).argsort()
            test_rank_list.append(int(np.squeeze(np.argwhere(cur_row_rank == j)))+1)
        #
        dict_cum_prob_at_k = {}
        test_rank_array = np.array(test_rank_list)
        num_test = float(len(test_rank_list))
        for k in np.arange(1,at_k+1):
            num_match = float(np.sum(test_rank_array == k))
            cur_prob = num_match/num_test
            print("k: ",k," num_match: ",num_match," num_test: ",num_test," cur_prob: ",cur_prob)
            if k == 1:
                dict_cum_prob_at_k[k] = cur_prob
            else:
                dict_cum_prob_at_k[k] = dict_cum_prob_at_k[k-1] + cur_prob
        #
        return dict_cum_prob_at_k

    def __get_recall_at_k(self,Rpred,Rtest,at_k):
        if self.is_val_transpose:
            Rpred = Rpred.T
        print("---")
        print("get_recall_at_"+str(at_k)+": ")
        print("Rpred.shape: ",Rpred.shape)
        print("Rtest.shape: ",Rtest.shape)
        num_rows = Rtest.shape[0]
        num_cols = Rtest.shape[1]
        dict_recall_at_k = {}
        dict_k_miss_user = {}
        for k in np.arange(1,at_k+1):
            recall_at_k_list = []
            num_rows_without_test_entries = 0
            for i in np.arange(num_rows):
                #pred
                cur_row_pred_idx_topk = np.argpartition(Rpred[i], -k)[-k:] #(-Rpred[i]).argsort()[:k]
                #target
                cur_row_target = Rtest[i]
                cur_row_target_idx = np.argwhere(cur_row_target).ravel()
                #recall
                num_match = len(set(cur_row_pred_idx_topk).intersection(cur_row_target_idx))
                num_total = len(cur_row_target_idx)
                if num_total > 0:
                    recall_at_k = num_match/float(num_total)
                    #if k == at_k:
                    #    print("k: ",k," i: ",i," num_match: ",num_match," num_total: ",num_total," recall_at_k: ",recall_at_k)
                    recall_at_k_list.append(recall_at_k)
                else:
                    #if k == at_k:
                    #    print("k: ",k," i: ",i," num_match: ",num_match," num_total: ",num_total," recall_at_k: NA")    
                    num_rows_without_test_entries+=1
            if num_rows_without_test_entries > 0:
                dict_k_miss_user[k] = num_rows_without_test_entries
            dict_recall_at_k[k] = np.mean(recall_at_k_list)
        #
        print("--debug start--")
        print("num_rows_without_test_entries:")
        for k in dict_k_miss_user.keys():
            print("k: ",k, " #users: ",dict_k_miss_user[k])
        print("--debug end--")
        return dict_recall_at_k

    def __get_val_performance(self,Rpred,Rtest):
        val_perf = None
        if self.val_metric == "rmse":
            val_perf = self.__get_rmse(Rpred,Rtest)
        elif self.val_metric == "r@k":
            val_perf = self.__get_recall_at_k(Rpred,Rtest,self.at_k)
        elif self.val_metric == "p@k":
            val_perf = self.__get_prob_at_k(Rpred,Rtest,self.at_k)
        elif self.val_metric == "auc":
            val_perf = self.__get_auc(Rpred,Rtest)
        else:
            assert False, "Unknown best parameter selection criterion:"+str(self.val_metric)
        return val_perf

    def fit(self):
        #dcmf model construction
        print("dcmf - model construction - start")
        self.__input_transformation()
        self.__network_construction()
        print("dcmf - model construction - end")
        print("#")
        if self.is_pretrain:
            self.__pretrain() 
        print("#")
        print("dcmf.fit - start")
        #opt algo setup
        criterion = torch.nn.MSELoss()
        model_params = []
        for e_id in self.G.keys():
            params_temp = list(self.N_aec_dict[e_id].parameters())
            #print("aec for e_id: ",e_id,", #params: ",len(params_temp))
            model_params+=params_temp
        optimizer = torch.optim.Adam(model_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        #main loop - training - start
        self.dict_epoch_loss = {}
        self.dict_epoch_aec_rec_loss = {}
        self.dict_epoch_mat_rec_loss = {}
        epoch = 1
        prev_loss_epoch = 0
        while True:
            if epoch > self.max_epochs:
                break
            #epoch - start
            s = time.time()
            #init - chunks prev idx (bookeeping)
            C_chunk_idx_prev_dict = {}
            for e_id in self.G.keys():
                C_chunk_idx_prev_dict[e_id] = 0
            #init - holds factors' chunks learnt (used for final matrix reconstruction after training)
            U_chunks_list_dict = {}
            for e_id in self.G.keys():
                U_chunks_list_dict[e_id]=[]
            #init - chunks to epoch loss
            loss_epoch = 0
            loss_aec_rec_epoch = {}
            for e_id in self.G.keys():
                loss_aec_rec_epoch[e_id] = 0
            loss_mat_rec_epoch = {}
            for X_id in self.X_data_var.keys():
                loss_mat_rec_epoch[X_id] = 0
            #chunks processing - start
            for i in np.arange(self.num_chunks):
                #load current batch data 
                C_chunk_batch = {}
                for e_id in self.G.keys():
                    C_chunk_batch[e_id] = self.C_dict_chunks[e_id][i] #load current batch input
                #update - chunks idx
                C_chunk_idx_dict = {}
                for e_id in self.G.keys():
                    C_chunk_idx_dict[e_id] = C_chunk_idx_prev_dict[e_id] + C_chunk_batch[e_id].shape[0]
                #train
                #autoencoder reconstruction - chunk
                C_chunk_rec_dict = {}
                U_chunk_dict = {}
                for e_id in self.G.keys():
                    C_chunk_rec_dict[e_id],U_chunk_dict[e_id] = self.N_aec_dict[e_id](C_chunk_batch[e_id])
                    #update - factors' chunks learnt
                    temp_chunks_list = U_chunks_list_dict[e_id]
                    temp_chunks_list.append(U_chunk_dict[e_id])
                    U_chunks_list_dict[e_id] = temp_chunks_list
                #matrix reconstruction - chunk
                X_data_var_chunk_rec = {}
                for X_id in self.X_data_var.keys():
                    rc_tuple = self.X_meta[X_id]
                    row_entity = rc_tuple[0]
                    col_entity = rc_tuple[1]
                    U_chunk_row = U_chunk_dict[row_entity]
                    U_chunk_col = U_chunk_dict[col_entity]
                    X_chunk_rec = U_chunk_row.mm(U_chunk_col.transpose(1,0))
                    X_data_var_chunk_rec[X_id] = X_chunk_rec
                #loss
                #autoencoder reconstruction loss - chunk
                loss_aec_rec_dict = {}
                for e_id in C_chunk_batch.keys():
                    loss_aec_rec_dict[e_id] = criterion(C_chunk_rec_dict[e_id],C_chunk_batch[e_id])
                #matrix reconstruction loss - chunk
                loss_mat_rec_dict = {}
                for X_id in self.X_data_var.keys():
                    X_temp = self.X_data_var[X_id]
                    rc_tuple = self.X_meta[X_id]
                    row_entity = rc_tuple[0]
                    col_entity = rc_tuple[1]
                    X_chunk = X_temp[C_chunk_idx_prev_dict[row_entity]:C_chunk_idx_dict[row_entity],\
                                     C_chunk_idx_prev_dict[col_entity]:C_chunk_idx_dict[col_entity]]
                    loss_mat_rec_dict[X_id] = criterion(X_data_var_chunk_rec[X_id],X_chunk)
                #sum all losses
                loss = np.sum(list(loss_aec_rec_dict.values())) + np.sum(list(loss_mat_rec_dict.values()))
                #backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                #update - loss info
                loss_epoch += loss.item()
                for e_id in self.G.keys():
                    loss_aec_rec_epoch[e_id] += loss_aec_rec_dict[e_id].item()
                for X_id in self.X_data_var.keys():
                    loss_mat_rec_epoch[X_id] += loss_mat_rec_dict[X_id].item()
                #update - chunks prev idx
                for e_id in self.C_dict_chunks.keys():
                    C_chunk_idx_prev_dict[e_id] = C_chunk_idx_dict[e_id]
            #chunks processing - end
            self.dict_epoch_loss[epoch] = loss_epoch
            self.dict_epoch_aec_rec_loss[epoch] = loss_aec_rec_epoch
            self.dict_epoch_mat_rec_loss[epoch] = loss_mat_rec_epoch
            #epoch - end
            e = time.time()
            print("epoch: ",epoch," total loss L: ",loss_epoch," Took ",round(e-s,1)," secs.")
            #update - counter
            epoch+=1
            if self.__is_converged(prev_loss_epoch,loss_epoch,self.convg_thres):
                print("**train converged**")
                break
            prev_loss_epoch = loss_epoch
        #main loop - training - end
        #list of losses, loss/elements of the list corresponds to tasks
        self.loss_list =  list(loss_aec_rec_epoch.values()) + list(loss_mat_rec_epoch.values())
        #
        #Build U_dict_ - entity representations
        for e_id in U_chunks_list_dict.keys():
            self.U_dict_[e_id] = torch.cat(U_chunks_list_dict[e_id])
        #Build X_prime_dict_ - matrix reconstructions
        for X_id in self.X_meta.keys():
            rc_tuple = self.X_meta[X_id]
            row_entity = rc_tuple[0]
            col_entity = rc_tuple[1]
            self.X_prime_dict_[X_id] = self.U_dict_[row_entity].mm(self.U_dict_[col_entity].transpose(1,0))
        #
        for X_id in self.X_val.keys():
            self.X_val_perf[X_id] = self.__get_val_performance(self.X_prime_dict_[X_id].cpu().data.numpy(),self.X_val[X_id])
        print("#")
        print("dcmf.fit - end")
  
