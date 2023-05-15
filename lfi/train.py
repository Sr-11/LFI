import sys
# sys.path.append(sys.path[0]+"/..")
# sys.path.append(sys.path[0])
import importlib.util
import numpy as np
import torch
from .utils import *
from matplotlib import pyplot as plt
# import pickle
import torch.nn as nn
# import time
import pandas as pd
# import cProfile
import os
# from GPUtil import showUtilization as gpu_usage
# from numba import cuda
from tqdm import tqdm
import os
import gc
# import config
# import global_config
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
torch.backends.cudnn.deterministic = True
dtype = torch.float
device = torch.device("cuda:0")
np.random.seed(42)
torch.manual_seed(42)

# define the pre-processing function
def pre_process(dataset_P,dataset_Q, n, batch_size):
    batches = (n-1)//batch_size + 1 # last batch could be empty
    n = batches*batch_size  
    X = dataset_P[0:n]
    Y = dataset_Q[0:n]
    total_S = [(X[i*batch_size:(i+1)*batch_size], 
                Y[i*batch_size:(i+1)*batch_size]) 
                for i in range(batches)]
    total_S = [MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in total_S]
    return total_S

# define the early stopping criterion
def early_stopping(validation_losses, patience):
    return len(validation_losses) - np.argmin(validation_losses) > patience
    
# train
def train(n_tr, total_S, S_validate, kernel, optimizer, N_epoch, save_every, path, patience, **kwargs):  
    # validation records
    J_validation_records = []
    train_loss_records = []
    # start training
    for t in range(N_epoch):
        # run one epoch
        print('----- n_tr =', n_tr, ', epoch =', t, '-----')
        order = np.random.permutation(len(total_S))
        for ind in tqdm(order):
            optimizer.zero_grad()
            obj = kernel.compute_loss(total_S[ind])
            obj.backward()
            optimizer.step()   
            train_loss_records.append(obj.item())
        # validation
        J_validation_records.append( kernel.compute_loss(S_validate, require_grad=False).item() )
        print('validation =', J_validation_records[-1])
        gc.collect()
        torch.cuda.empty_cache()
        # print
        if t%save_every == 0:
            torch.save(kernel, path+'/kernel.pt')
            plt.plot(J_validation_records)
            plt.savefig(path+'/validations_epoch.png')
            plt.clf()
            plt.plot(train_loss_records)
            plt.savefig(path+'/train_loss_iter.png')
            plt.clf()
        # early stopping
        if early_stopping(J_validation_records, patience):
            break
    torch.save(kernel, path+'/kernel.pt')
    return kernel, J_validation_records

def main(config_path):
    # import config
    print('config_path =', config_path)
    config_spec = importlib.util.spec_from_file_location('config', config_path)
    config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config)
    # import neural model
    model_path = config.model_configs['model_path']
    model_dir = os.path.dirname(os.path.realpath(model_path))
    sys.path.append(model_dir)
    from model import Model
    # import data
    os.environ["CUDA_VISIBLE_DEVICES"]= config.train_param_configs['gpu_id'] 
    dataset = np.load(config.resource_configs['Higgs_path'])
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5170877, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5829122, 28) 
    # prepare n_list
    n_org_list = config.train_param_configs['n_tr_list']
    repeats = config.train_param_configs['repeats']
    n_tr_list = []
    for n in n_org_list:
        for i in range(repeats):
            n_tr_list.append(n+i)
    # other parameters
    batch_size = config.train_param_configs['batch_size']
    N_epoch = config.train_param_configs['N_epoch']
    checkpoints_path = config.expr_configs['checkpoints_path']
    validate_size = config.train_param_configs['validation_size']
    learning_rate = config.train_param_configs['learning_rate']
    momentum = config.train_param_configs['momentum']
    save_every = config.train_param_configs['save_every']
    patience = config.train_param_configs['patience']
    # train
    for n in n_tr_list:
        print('----------- TRAIN n_tr = %d ----------'%n)
        total_S = pre_process(dataset_P, dataset_Q, n, batch_size)
        S_validate = MatConvert(np.concatenate((dataset_P[n+np.random.choice(n, validate_size, replace=False)], 
                            dataset_Q[n+np.random.choice(n, validate_size, replace=False)]), axis=0), device, dtype)
        kernel = Model()
        path = checkpoints_path + '/n_tr=%d'%n
        optimizer = torch.optim.SGD(kernel.params, lr=learning_rate, momentum=momentum)
        if not os.path.exists(path):
            os.makedirs(path)
        kernel, J = train(n, total_S, S_validate, 
                          kernel, optimizer, 
                          N_epoch, save_every, path, patience)
        
if __name__ == "__main__":
    config_path = sys.argv[1]
    main(config_path)