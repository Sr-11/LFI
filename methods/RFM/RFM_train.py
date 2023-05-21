import numpy as np
import torch
from matplotlib import pyplot as plt
import os, sys, inspect, gc
import config as config
from rfm import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
torch.backends.cudnn.deterministic = True
dtype =	 torch.float
device = torch.device("cuda:0")
torch.manual_seed(42)
np.random.seed(42)
current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))) 

# define the early stopping criterion
def early_stopping(validation_losses, patience):
    return len(validation_losses) - np.argmin(validation_losses) >= patience

def main(**kwargs): 
    # load config
    n_tr_list = config.train_param_configs['n_tr_list']
    repeat = config.train_param_configs['repeats']
    batch_size = config.train_param_configs['batch_size']
    N_epoch = config.train_param_configs['N_epoch']
    patience = config.train_param_configs['patience']
    os.environ["CUDA_VISIBLE_DEVICES"]= config.train_param_configs['gpu']
    median_heuristic = config.train_param_configs['median_heuristic']
    if 'gpu' in kwargs:
        os.environ["CUDA_VISIBLE_DEVICES"]= kwargs['gpu']
    if 'n_tr_list' in kwargs:
        n_tr_list = kwargs['n_tr_list']
    if 'repeat' in kwargs:
        repeat = kwargs['repeat']
    if 'patience' in kwargs:
        patience = kwargs['patience']
    if 'median_heuristic' in kwargs:
        median_heuristic = kwargs['median_heuristic']
    checkpoints_path = os.path.join(current_dir, 'checkpoints')
    # load data
    dataset = np.load(config.resource_configs['Higgs_path'])
    dataset = torch.from_numpy(dataset).to(device=device, dtype=dtype)
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5170877, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5829122, 28)
    # trian
    zero_in_gpu = torch.zeros(1).to(device=device, dtype=dtype)
    one_in_gpu = torch.ones(1).to(device=device, dtype=dtype)
    for n_tr in n_tr_list:
        for r in repeat:
            print('------ n =', n_tr, '------')
            validate_size = int(np.sqrt(n_tr))
            # pre-process
            trainset = []
            rand_order = np.random.permutation(n_tr)
            for i in rand_order:
                trainset.append((dataset_P[i], zero_in_gpu)); trainset.append((dataset_Q[i], one_in_gpu))
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
            test_loader = ( torch.cat([dataset_P[n_tr:n_tr+validate_size], dataset_Q[n_tr:n_tr+validate_size]]), 
                            torch.cat([torch.zeros(validate_size),torch.ones(validate_size)]).to(device=device, dtype=dtype).squeeze() )
            # make dir
            checkpoint_dir = os.path.join(checkpoints_path, 'n_tr=%d#%d'%(n_tr,r))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # train
            M, explode_flag = rfm(train_loader, test_loader, 
                    N_epoch=N_epoch,
                    device=device, dtype=dtype,
                    checkpoint_path=checkpoint_dir,
                    patience=patience,
                    early_stopping=early_stopping,
                    median_heuristic=median_heuristic,)
            gc.collect(); torch.cuda.empty_cache()
            # explode_flag
            if explode_flag:
                print('!! exploded, append again rfm !!')
                n_tr_list.append(n_tr)

if __name__ == "__main__":
    main(**dict([arg.split('=') for arg in sys.argv[1:]]))