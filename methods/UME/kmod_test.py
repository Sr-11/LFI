import numpy as np
import torch
import sys
from UME_utils import *
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
import UME_config as UME_config
import numpy as np
import UME_config as UME_config
import sys
import pickle

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= UME_config.test_param_configs['gpu_id'] # specify which GPU(s) to be used
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
                    folder_path='', epoch=None,
                    method='UME', 
                    repeats=10,
                    test_soft=True, test_hard=True):
    V, kernel = load_model(folder_path, epoch=epoch)
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
        if test_soft:
            p_soft_list[j] = get_pval_from_evaluated_scores(X_scores, Y_scores, 
                                                norm_or_binom=True, thres=None, verbose = False)
            gc.collect()
            torch.cuda.empty_cache()
        if test_hard:
            X_test_scores = kernel.compute_scores(X_test, Y_test, X_test, V) # (n_ev,)
            Y_test_scores = kernel.compute_scores(X_test, Y_test, Y_test, V) # (n_ev,)
            X_test_scores = X_test_scores.cpu()
            Y_test_scores = Y_test_scores.cpu()
            thres, _, _ = get_thres_from_evaluated_scores(X_test_scores, Y_test_scores)
            p_hard_list[j] = get_pval_from_evaluated_scores(X_scores, Y_scores, 
                                                norm_or_binom=False, thres=thres, verbose = False)
            del X_test_scores, Y_test_scores
            gc.collect()
            torch.cuda.empty_cache()
        del X_test, Y_test, X_eval, Y_eval, X_scores, Y_scores
        gc.collect()
        torch.cuda.empty_cache()
        # save hist
        # plt.hist(X_scores.cpu().detach().numpy(), bins=100, alpha=0.5, label='X')
        # plt.hist(Y_scores.cpu().detach().numpy(), bins=100, alpha=0.5, label='Y')
        # plt.legend(loc='upper right')
        # plt.savefig(folder_path+'hist.png')
    return p_soft_list, p_hard_list

def package_pval_data():
    ns = UME_config.test_param_configs['n_tr_list']
    pval_mid_path = UME_config.expr_configs['pval_mat_path']

    num_models = UME_config.test_param_configs['num_models']
    num_repeats = UME_config.test_param_configs['num_repeats']

    pval_dict = {}
    pval_dict['UME_soft'] = {}
    pval_dict['UME_hard'] = {}

    for n_tr in ns:
        print("------------------- n_tr = %d -------------------"%n_tr)
        p_soft_mat = np.load(sys.path[0]+pval_mid_path+'/n_tr=%d_soft.npy'%n_tr)
        p_hard_mat = np.load(sys.path[0]+pval_mid_path+'/n_tr=%d_hard.npy'%n_tr)
        assert p_soft_mat.shape == p_hard_mat.shape == (num_models, num_repeats)
        pval_dict['UME_soft'][n_tr] = p_soft_mat
        pval_dict['UME_hard'][n_tr] = p_hard_mat

    with open(sys.path[0]+pval_mid_path+'/pval_dict.pkl', 'wb') as f:
        pickle.dump(pval_dict, f, pickle.HIGHEST_PROTOCOL)


dataset = np.load(UME_config.resource_configs['Higgs_path'])
dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)
dataset_P = MatConvert(dataset_P, device=device, dtype=dtype)
dataset_Q = MatConvert(dataset_Q, device=device, dtype=dtype)

    # num_models = 10 # change trained
def run_test():
    # num_repeats = 10 # change sample
    # ns = np.array([1300000, 1000000, 700000, 400000, 200000, 50000])
    ns = UME_config.test_param_configs['n_tr_list']
    num_models = UME_config.test_param_configs['num_models']
    num_repeats = UME_config.test_param_configs['num_repeats']

    n_ev = UME_config.test_param_configs['n_ev']
    n_te = UME_config.test_param_configs['n_te']

    pval_path = UME_config.expr_configs['pval_mat_path']
    chekpoints_path = UME_config.expr_configs['checkpoints_path']

    test_soft = UME_config.test_param_configs['test_soft']
    test_hard = UME_config.test_param_configs['test_hard']

    for n_tr in ns:
        print("------------------- n_tr = %d -------------------"%n_tr)
        p_soft_mat = np.zeros((num_models,num_repeats)) # p_soft_mat[i,j] = model i and test j
        p_hard_mat = np.zeros((num_models,num_repeats))
        for i in trange(num_models):
            n_tr_label = n_tr+i
            folder_path = chekpoints_path+'/checkpoints n_tr=%d/'%n_tr_label
            with open(folder_path+'data.pickle', 'rb') as file:
                data = pickle.load(file); epoch = data['epoch']
            p_soft_mat[i,:], p_hard_mat[i,:] = simulate_p_value(n_tr_label, n_ev, min(n_tr,n_te),
                                                                folder_path=folder_path, epoch = epoch,
                                                                method='UME', 
                                                                repeats=num_repeats,
                                                                test_soft=test_soft, test_hard=test_hard)
        try:
            os.mkdir(pval_path)
        except:
            pass
        if test_soft:
            np.save(pval_path+'/n_tr=%d_soft.npy'%n_tr, p_soft_mat)
        if test_hard:
            np.save(pval_path+'/n_tr=%d_hard.npy'%n_tr, p_hard_mat)
        gc.collect()
        torch.cuda.empty_cache()
        clear_output(wait=True)

    # package_pval_data()

if __name__ == "__main__":
    run_test()