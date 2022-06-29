import numpy as np
import collections
import sys
#
import torch
from torch import nn

class autoencoder(nn.Module):
   
    def get_actf(self,actf_name):
        if actf_name is "relu":
            A = nn.ReLU()
        elif actf_name is "sigma":
            A = nn.Sigmoid()
        elif actf_name is "tanh":
            A = nn.Tanh()
        elif actf_name is "lrelu":
            A = nn.LeakyReLU()
        else:
            print("Unknown activation function: ",actf_name)
            sys.exit(1)
        return A
    
    def get_encoder(self):
        return self.encoder
    
    def get_encoder_params(self):
        params_list = []
        for temp in self.encoder.parameters():
            params_list.append(temp.cpu().data.numpy())
        return params_list        
    
    def get_aec_params(self):
        params_list = []
        for temp in self.parameters():
            params_list.append(temp.cpu().data.numpy())
        return params_list

    def __init__(self,input_dim,k_list,actf_list,\
                        is_linear_last_enc_layer=False,is_linear_last_dec_layer=False,params_list=None): #,input_dim,k_list):
        super(autoencoder, self).__init__()
        temp_params_list = []
        if params_list is not None:
            #print("Initializing aec with pretrained parameters...")
            for param in params_list:
                temp_params_list.append(param)
        #else:
            #print("Initializing aec with random parameters...")
        #
        #print("input_dim: ",input_dim)
        #print("k_list: ",k_list)
        #print("actf_list: ",actf_list)
        #encoding layers
        enc_layers_dict = collections.OrderedDict()
        temp_k_decode = []
        num_enc_layers = len(k_list)
        k1 = input_dim
        l=0
        for i in np.arange(num_enc_layers):
            k2 = k_list[i]
            temp_layer = nn.Linear(int(k1), int(k2))
            if params_list is not None:
                #print("init w: ",temp_layer.weight.data.shape," - ",temp_params_list[l].data.shape)
                temp_layer.weight.data = temp_params_list[l].data
                l+=1
                #print("init b: ",temp_layer.bias.data.shape," - ",temp_params_list[l].data.shape)
                temp_layer.bias.data = temp_params_list[l].data
                l+=1
            enc_layers_dict["enc-"+str(i)] = temp_layer
            if not is_linear_last_enc_layer:
                enc_layers_dict["act-"+str(i)] = self.get_actf(actf_list[i]) 
            else:
                if i != (num_enc_layers-1):
                    enc_layers_dict["act-"+str(i)] = self.get_actf(actf_list[i]) 
            temp_k_decode.append((k1,k2)) 
            k1 = k2
        #decoding layers
        dec_layers_dict = collections.OrderedDict()
        temp_k_decode.reverse()
        for k_tup in temp_k_decode:
            i+=1
            k1 = k_tup[1]
            k2 = k_tup[0]
            temp_layer = nn.Linear(int(k1), int(k2))
            if params_list is not None:
                #print("init w: ",temp_layer.weight.data.shape," - ",temp_params_list[l].data.shape)
                temp_layer.weight.data = temp_params_list[l].data
                l+=1
                #print("init b: ",temp_layer.bias.data.shape," - ",temp_params_list[l].data.shape)
                temp_layer.bias.data = temp_params_list[l].data
                l+=1
            dec_layers_dict["dec-"+str(i)] = temp_layer
            if not is_linear_last_dec_layer:
                dec_layers_dict["act-"+str(i)] = self.get_actf(actf_list[i])
            else:
                if (i != len(actf_list)-1):
                    dec_layers_dict["act-"+str(i)] = self.get_actf(actf_list[i])
        #
        self.encoder = nn.Sequential(enc_layers_dict)
        self.decoder = nn.Sequential(dec_layers_dict)
        #
        # print("encoder: ")
        # print(self.encoder)
        # print("decoder: ")
        # print(self.decoder)

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_dec,x_enc
