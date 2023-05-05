import numpy as np
import torch
from matplotlib import pyplot as plt
import os
from tqdm import tqdm, trange
# import autograd.numpy as np
import pickle
import sys
import config
from rfm import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= config.train_param_configs['gpu_id'] # specify which GPU(s) to be used
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
 
torch.backends.cudnn.deterministic = True
dtype = torch.float
device = torch.device("cuda:0")
torch.manual_seed(42)
np.random.seed(42)

debug = False

# define the early stopping criterion
def early_stopping(validation_losses, epoch):
    i = np.argmin(validation_losses)
    # print(i)
    if epoch - i > 10:
        return True
    else:
        return False

# define the pre-processing function
def pre_process(torchset,n_samples,num_classes=10, normalize=False):
    indices = list(np.random.choice(len(torchset),n_samples))
    trainset = []
    for ix in indices:
        x,y = torchset[ix]
        ohe_y = torch.zeros(num_classes)
        ohe_y[y] = 1
        if normalize:
            trainset.append(((x/np.linalg.norm(x)).reshape(-1),ohe_y))
        else:
            trainset.append((x.reshape(-1),ohe_y))
    return trainset

if __name__ == "__main__":
    dataset = np.load(config.resource_configs['Higgs_path'])
    print('signal : background =',np.sum(dataset[:,0]),':',dataset.shape[0]-np.sum(dataset[:,0]))
    print('signal :',np.sum(dataset[:,0])/dataset.shape[0]*100,'%')
    # split into signal and background and move to gpu
    dataset = torch.from_numpy(dataset).to(device=device, dtype=dtype)
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5170877, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5829122, 28) 
    n_org_list = config.train_param_configs['n_tr_list']
    repeats = config.train_param_configs['repeats']
    n_tr_list = []
    for n in n_org_list:
        for i in range(repeats):
            n_tr_list.append(n+i)

    batch_size = config.train_param_configs['batch_size']
    N_epoch = config.train_param_configs['N_epoch']
    checkpoints_path = config.expr_configs['checkpoints_path']
    zero_in_gpu = torch.zeros(1).to(device=device, dtype=dtype)
    one_in_gpu = torch.ones(1).to(device=device, dtype=dtype)
    
    for n_tr in n_tr_list:
        print('------ n =', n_tr, '------')
        # pre-process
        trainset = []
        for i in range(n_tr):
            trainset.append((dataset_P[i], zero_in_gpu))
            trainset.append((dataset_Q[i], one_in_gpu))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        validateset = []
        for i in range(10000):
            validateset.append((dataset_P[n_tr+i], zero_in_gpu))
            validateset.append((dataset_Q[n_tr+i], one_in_gpu))
        test_loader = torch.utils.data.DataLoader(validateset, batch_size=batch_size, shuffle=False)
        # run rfm
        try: #make dir
            os.mkdir(checkpoints_path+'/n_tr=%d'%n_tr)
        except:
            pass
        M = rfm(train_loader, test_loader, iters=N_epoch, loader=True, classif=False, device=device, 
                   checkpoint_path=checkpoints_path+'/n_tr=%d'%n_tr,
                   early_stopping=early_stopping)
        