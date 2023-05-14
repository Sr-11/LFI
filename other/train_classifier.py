import numpy as np
import torch
import sys
from utils import *
from matplotlib import pyplot as plt
import pickle
import torch.nn as nn
import time
import pandas as pd
import cProfile
import os
from GPUtil import showUtilization as gpu_usage
from numba import cuda
from tqdm import tqdm, trange
import os
import gc
import config
import sys
sys.path.append(sys.path[0]+"/..")
import global_config

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= config.train_param_configs['gpu_id'] 
torch.backends.cudnn.deterministic = True
dtype = torch.float
device = torch.device("cuda:0")
torch.manual_seed(42)
np.random.seed(42)

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
def early_stopping(validation_losses):
    return len(validation_losses) - np.argmin(validation_losses) > 10
    
# train
def train(n_tr, total_S, S_validate, Model, optimizer, N_epoch, save_every, path, **kwargs):  
    # validation records
    J_validation_records = []
    train_loss_records = []
    # start training
    for t in range(N_epoch):
        # run one epoch
        print('----- epoch =', t, '-----')
        order = np.random.permutation(len(total_S))
        for ind in tqdm(order, leave=False):
            optimizer.zero_grad()
            obj = Model.compute_loss(total_S[ind], method=kwargs['method'])
            obj.backward()
            optimizer.step()   
            train_loss_records.append(obj.item())
            # if i % 100 == 0:
        X_scores = Model.compute_scores(X_calibrate, require_grad=False)
        Y_scores = Model.compute_scores(Y_calibrate, require_grad=False)
        plot_hist(X_scores, Y_scores, path+'/hist', 'epoch %d'%t)
        # validation
        J_validation_records.append( Model.compute_loss(S_validate, require_grad=False, method=kwargs['method']).item() )
        print('validation =', J_validation_records[-1])
        gc.collect()
        torch.cuda.empty_cache()
        # print
        if t%save_every == 0:
            torch.save(Model, path+'/Model.pt')
            # Model.save_checkpoint(path)
            plt.plot(J_validation_records)
            plt.savefig(path+'/J_validations_T.png')
            plt.clf()
            plt.plot(train_loss_records)
            plt.savefig(path+'/train_loss_T.png')
            plt.clf()
        # early stopping
        if early_stopping(J_validation_records):
            break

    torch.save(Model, path+'/Model.pt')
    return Model, J_validation_records

if __name__ == "__main__":
    # load data, please use .npy ones (40000), .gz (10999999)(11e6) is too large.
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
    # batch_size = config.train_param_configs['batch_size']
    batch_size = 100001
    N_epoch = config.train_param_configs['N_epoch']
    checkpoints_path = config.expr_configs['checkpoints_path']
    # validate_size = config.train_param_configs['validation_size']
    validate_size = 100001
    # learning_rate = config.train_param_configs['learning_rate']
    learning_rate = 0.01
    momentum = config.train_param_configs['momentum']
    save_every = config.train_param_configs['save_every']

    # train
    n_sys_input = int(sys.argv[1])
    method = sys.argv[2]
    print('n_tr=%d, method=%s'%(n_sys_input, method))
    for n in [n_sys_input]:
        print('----------- n_tr = %d ----------'%n)
        total_S = pre_process(dataset_P, dataset_Q, n, batch_size)
        S_validate = MatConvert(np.concatenate((dataset_P[n+np.random.choice(n, validate_size, replace=False)], 
                            dataset_Q[n+np.random.choice(n, validate_size, replace=False)]), axis=0), device, dtype)
        
        X_eval_validate = dataset_P[n+np.random.choice(dataset_P.shape[0]-n, validate_size, replace=False)]
        Y_eval_validate = dataset_Q[n+np.random.choice(dataset_Q.shape[0]-n, validate_size, replace=False)]
        X_calibrate = dataset_P[n+np.random.choice(dataset_P.shape[0]-n, validate_size, replace=False)]
        Y_calibrate = dataset_Q[n+np.random.choice(dataset_Q.shape[0]-n, validate_size, replace=False)]
        X_eval_validate = MatConvert(X_eval_validate, device, dtype)
        Y_eval_validate = MatConvert(Y_eval_validate, device, dtype)
        X_calibrate = MatConvert(X_calibrate, device, dtype)
        Y_calibrate = MatConvert(Y_calibrate, device, dtype)
        
        Model = Classifier()
        path = checkpoints_path + '/n_tr=%d_'%n + method
        # kernel = torch.load(path+'/kernel.pt')
        optimizer = torch.optim.SGD(Model.params, lr=learning_rate, momentum=momentum)
        # optimizer = torch.optim.Adam(Model.params)
        if not os.path.exists(path):
            os.makedirs(path)
        Model, J = train(n, total_S, S_validate, 
                          Model, optimizer, 
                          N_epoch, save_every, path,
                          X_eval_validate=X_eval_validate,
                          Y_eval_validate=Y_eval_validate,
                          X_calibrate=X_calibrate,
                          Y_calibrate=Y_calibrate,
                          method=method)