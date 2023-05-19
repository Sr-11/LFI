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
from IPython.display import clear_output
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
    print(patience)
    return len(validation_losses) - np.argmin(validation_losses) >= patience

# find epoch-patience checkpoint
def maintain_checkpoints(dir, patience, kernel, kernel_histories):
    if patience != 100:
        torch.save(kernel, dir+'/kernel-0.pt')
        kernel_copy = torch.load(dir+'/kernel-0.pt')
        kernel_histories.append(kernel_copy)
        assert len(kernel_histories) == patience+1
        for i in range(0, patience+1):
            if kernel_histories[patience-i] != None:
                torch.save(kernel_histories[patience-i], dir+'/kernel_-%d.pt'%(i))
        return kernel_histories.pop(0)
    else:
        return kernel
# train
def train(n_tr, total_S, S_validate, kernel, optimizer, N_epoch, save_every, path, patience, **kwargs):  
    # n_te = 10000
    # n_ev = 10000
    # X_test = kwargs['dataset_P'][ np.random.choice(n_tr, n_te, replace=False) ]
    # Y_test = kwargs['dataset_Q'][ np.random.choice(n_tr, n_te, replace=False) ]
    # X_eval = kwargs['dataset_P'][ n_tr + np.random.choice(kwargs['dataset_P'].shape[0]-n_tr, n_ev, replace=False) ]
    # Y_eval = kwargs['dataset_Q'][ n_tr + np.random.choice(kwargs['dataset_Q'].shape[0]-n_tr, n_ev, replace=False) ]
    # X_test = MatConvert(X_test, device, dtype)
    # Y_test = MatConvert(Y_test, device, dtype)
    # X_eval = MatConvert(X_eval, device, dtype)
    # Y_eval = MatConvert(Y_eval, device, dtype)
    # validation records
    J_validation_records = []
    train_loss_records = []
    kernel_histories = [None]*patience
    # start training
    for t in range(N_epoch):
        # run one epoch
        clear_output(wait=True)
        print('----- n_tr =', n_tr, ', epoch =', t, '-----')
        order = np.random.permutation(len(total_S))
        for ind in tqdm(order):
            optimizer.zero_grad()
            obj = kernel.compute_loss(total_S[ind])
            obj.backward()
            optimizer.step()   
            train_loss_records.append(obj.item())
        kernel.epoch += 1
        # validation
        J_validation_records.append( kernel.compute_loss(S_validate, require_grad=False).item() )
        print('validation =', J_validation_records[-1])
        gc.collect()
        torch.cuda.empty_cache()
        # early stopping
        stop_flag = early_stopping(J_validation_records, patience)
        last_patience_kernel = maintain_checkpoints(path, patience, kernel, kernel_histories)
        # print
        save_every = 10
        if t%save_every == 0 or stop_flag == True:
            # torch.save(kernel, path+'/kernel.pt')
            plt.plot(J_validation_records)
            plt.savefig(path+'/validations_epoch.png')
            plt.clf()
            plt.plot(train_loss_records)
            plt.savefig(path+'/train_loss_iter.png')
            plt.clf()
            plt.close()
            plt.close('all')
        if stop_flag:
            torch.save(last_patience_kernel, path+'/kernel.pt')
            # with torch.no_grad():

            #     X_scores = kernel.compute_scores(X_test, Y_test, X_eval) # (n_ev,)
            #     Y_scores = kernel.compute_scores(X_test, Y_test, Y_eval) # (n_ev,)

            #     # plot
            #     if plot_hist_path != None:
            #         print("plotting histogram to %s"%(path+'/hist.png'))
            #         plot_hist(X_scores, Y_scores, path+'/hist.png', title='train')
            return last_patience_kernel, J_validation_records
    
    torch.save(kernel, path+'/kernel.pt')
    return kernel, J_validation_records

def main(config_path, **kwargs):
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
    median_flag = True
    # train
    # does kwargs['n_tr'] exists
    if 'n_tr_list' in kwargs:
        n_tr_list = kwargs['n_tr_list']
    if 'gpu' in kwargs:
        os.environ["CUDA_VISIBLE_DEVICES"]= kwargs['gpu']
    if 'median_flag' in kwargs:
        median_flag = kwargs['median_flag']
    if 'patience' in kwargs:
        patience = kwargs['patience']
    if 'checkpoints_path' in kwargs:
        checkpoints_path = kwargs['checkpoints_path']
    if 'batch_size' in kwargs:
        batch_size = kwargs['batch_size']
    n_eaise_error_list = []
    for n in n_tr_list:
        n_eaise_error_list.append(n)
        for _ in range(2):
            print('----------- TRAIN n_tr = %d ----------'%n)
            total_S = pre_process(dataset_P, dataset_Q, n, batch_size)
            S_validate = MatConvert(np.concatenate((dataset_P[n+np.random.choice(dataset_P.shape[0]-n, validate_size, replace=False)],  dataset_Q[n+np.random.choice(dataset_Q.shape[0]-n, validate_size, replace=False)]), axis=0), device, dtype)
            kernel = Model(median_heuristic_flag=median_flag, X_heu = total_S[0][:batch_size], Y_heu = total_S[0][batch_size:])
            path = checkpoints_path + '/n_tr=%d'%n
            optimizer = torch.optim.SGD(kernel.params, lr=learning_rate, momentum=momentum)
            if not os.path.exists(path):
                os.makedirs(path)
            kernel, Js = train(n, total_S, S_validate, 
                            kernel, optimizer, 
                            N_epoch, save_every, path, patience,
                            dataset_P=dataset_P, dataset_Q=dataset_Q)
            try:
                print('epoch:', kernel.epoch)
            except:
                pass
            if min(Js)<max(Js)-0.01:
                n_eaise_error_list.pop()
                break
            else:
                print('------------------------------------------')
                print('----- WARNING: training might failed -----')
                print('---- Check the loss-epoch plot in ckpt ----')
                print('------------------------------------------')
    
    if len(n_eaise_error_list)>0:
        print('Training might failed at n = ', n_eaise_error_list)

        
        
if __name__ == "__main__":
    config_path = sys.argv[1]
    main(config_path)