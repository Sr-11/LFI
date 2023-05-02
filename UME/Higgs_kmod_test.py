import numpy as np
import torch
import sys
from utils_UME import *
from matplotlib import pyplot as plt
import torch.nn as nn
import time
from numba import cuda
from tqdm import tqdm, trange
import os
import pyroc
import pandas as pd
import gc
from IPython.display import clear_output
import pickle
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  
device = torch.device("cuda:0")
dtype = torch.float32
torch.manual_seed(42)
np.random.seed(42)

class Tester():
    def __init__(self, X_te, Y_te, kernel, V):
        self.X_te = X_te
        self.Y_te = Y_te
        self.kernel = kernel
        self.V = V
    def compute_test_statistic(self, Z): # Z.shape = (m,d)
        # XZ = torch.cat((self.X_te, Z), dim=0)
        # YZ = torch.cat((self.Y_te, Z), dim=0)
        UME_pr = self.kernel.compute_UME_mean(self, self.X_te, Z, self.V)
        UME_qr = self.kernel.compute_UME_mean(self, self.Y_te, Z, self.V)
        return UME_pr - UME_qr

    def compute_test_statistics(self, Zs):# Zs.shape = (num,m,d)
        num = Zs.shape[0]
        stats = torch.zeros(num)
        for i in range(num):
             stats = self.compute_test_statistic(Zs[i])
        return stats


def simulate_p_value(n_tr, n_ev, n_te,
                    folder_path='', epoch=0, 
                    method='UME', 
                    repeats=10, 
                    batch_size=1024):
    V, kernel = load_model(folder_path, epoch=0)
    gc.collect()
    torch.cuda.empty_cache()
    #####################
    p_soft_list = np.zeros(repeats)
    p_hard_list = np.zeros(repeats)
    for j in range(repeats):
        idx = n_tr + np.random.choice(dataset_P.shape[0]-n_tr, n_ev, replace=False) 
        idy = n_tr + np.random.choice(dataset_Q.shape[0]-n_tr, n_ev, replace=False) 
        X_test = dataset_P[np.random.choice(n_tr, n_te, replace=False)]
        Y_test = dataset_Q[np.random.choice(n_tr, n_te, replace=False)]
        X_eval = dataset_P[idx]
        Y_eval = dataset_Q[idy]
        # Compute the test statistic on the test data
        X_scores = kernel.compute_scores(X_test, Y_test, X_eval, V) # (n_ev,)
        Y_scores = kernel.compute_scores(X_test, Y_test, Y_eval, V) # (n_ev,)
        gc.collect()
        p_soft_list[j] = get_pval_from_evaluated_scores(X_scores, Y_scores, 
                                              norm_or_binom=True, thres=None, verbose = False)
        p_hard_list[j] = 0

        # save hist
        plt.hist(X_scores.cpu().detach().numpy(), bins=100, alpha=0.5, label='X')
        plt.hist(Y_scores.cpu().detach().numpy(), bins=100, alpha=0.5, label='Y')
        plt.legend(loc='upper right')
        plt.savefig(folder_path+'hist.png')

    return p_soft_list, p_hard_list

if __name__ == "__main__":
    dataset = np.load('HIGGS_first_200_000.npy')
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)
    dataset_P = MatConvert(dataset_P, device=device, dtype=dtype)
    dataset_Q = MatConvert(dataset_Q, device=device, dtype=dtype)

    num_models = 10 # change trained
    num_repeats = 10 # change sample
    ns = np.array([1300000, 1000000, 700000, 400000, 200000, 50000])

    for n_tr in ns:
        p_soft_mat = np.zeros((num_models,num_repeats)) # p_soft_mat[i,j] = model i and test j
        p_hard_mat = np.zeros((num_models,num_repeats))
        for i in range(num_models):
            n_ev = min(n_tr,20000)
            n_te = 10000
            n_tr_label = n_tr+i
            folder_path = sys.path[0]+'/checkpoints n_tr=%d/'%n_tr_label
            with open(folder_path+'data.pickle', 'rb') as file:
                data = pickle.load(file); epoch = data['epoch']
            p_soft_mat[i,:], p_hard_mat[i,:] = simulate_p_value(n_tr_label, n_ev, n_te,
                                                                folder_path=folder_path, epoch=epoch, 
                                                                method='UME', 
                                                                repeats=num_repeats, 
                                                                batch_size=1024)
        np.save(sys.path[0]+'/pval_data/n_tr=%d_soft.npy'%n_tr, p_soft_mat)
        np.save(sys.path[0]+'/pval_data/n_tr=%d_hard.npy'%n_tr, p_hard_mat)
        clear_output(wait=True)