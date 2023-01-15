import numpy as np
import torch
import sys
from utils import Pdist2, MatConvert, load_model, compute_score_func
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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"  # specify which GPU(s) to be used

seed=42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
dtype = torch.float
device = torch.device("cuda:0")

H = 500
out= 100
L = 1
class DN(torch.nn.Module):
    def __init__(self):
        super(DN, self).__init__()
        self.restored = False
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28, H, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(H, out, bias=True),
        )
    def forward(self, input):
        output = self.model(input)
        return output

class another_DN(torch.nn.Module):
    def __init__(self):
        super(another_DN, self).__init__()
        self.restored = False
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, 28, bias=True),
        )
    def forward(self, input):
        output = self.model(input) + input
        return output

def get_pval(path):
    print('path is:', path)
    dataset = np.load('HIGGS.npy')
    # split into signal and background
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)

    model = DN().cuda()
    another_model = another_DN().cuda()
    model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst = load_model(model, another_model, path)

    # region
    # 一共选100*1100
    # N = 100
    # score_list = np.zeros(N)
    # for k in trange(N):
    #     background_events = dataset_P[np.random.choice(dataset_P.shape[0], 1000)]
    #     signal_events = dataset_Q[np.random.choice(dataset_Q.shape[0], 100)]
    #     Z = MatConvert(np.concatenate((signal_events, background_events), axis=0), device, dtype)
    #     Z_score = compute_score_func(Z, dataset_P, dataset_Q, 
    #                 model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, cst,
    #                 L=1, M=100)
    #     score_list[k] = torch.mean(Z_score)
    # print('score_list', score_list)
    # score = np.mean(score_list)
    # p_value = -(score-mean)/np.sqrt(var)
    # p_value_var = np.var((score_list-mean)/np.sqrt(var))
    # endregion

    # 选10000个P，算他们的分数
    X = MatConvert( dataset_P[np.random.choice(dataset_P.shape[0], 10000)], device, dtype)
    X_score = compute_score_func(X, dataset_P, dataset_Q, 
                    model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, cst,
                    L=1, M=10000).cpu().detach().numpy()
    Y = MatConvert( dataset_Q[np.random.choice(dataset_Q.shape[0], 10000)], device, dtype)
    Y_score = compute_score_func(Y, dataset_P, dataset_Q, 
                    model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, cst,
                    L=1, M=10000).cpu().detach().numpy()
    X_mean = np.mean(X_score)
    X_std = np.std(X_score)
    Y_mean = np.mean(Y_score)
    Y_std = np.std(Y_score)
    print('#datapoints =', len(X_score))
    print('X_mean =', X_mean)
    print('X_std =', X_std)
    print('Y_mean =', Y_mean)
    print('Y_std =', Y_std)
    # 直接算平均的P的分数和方差，平均的Q的分数，然后加权

    plt.hist(Y_score, bins=200, label='score_Q', alpha=0.5)
    plt.hist(X_score, bins=200, label='score_P', alpha=0.5)
    plt.legend()
    plt.savefig('score.png')

    Z_score = (10*X_mean + Y_mean)/11
    p_value = (Z_score-X_mean)/X_std
    print('----------------------------------')
    print('p_value =', p_value)
    print('----------------------------------')
    return p_value

if __name__ == "__main__":
    n = int(sys.argv[1])
    epoch = int(sys.argv[2])
    scheffe = int(sys.argv[3]) # 1 for scheffe, 0 for kernel
    path = './checkpoint%d/'%n+str(epoch)+'/'
    dtype = torch.float
    device = torch.device("cuda:0")

    # set seed
    seed=42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # get p-value
    path = './checkpoint%d/'%n+str(epoch)+'/'

    p_value = get_pval(path)

    print('p_value =', p_value)