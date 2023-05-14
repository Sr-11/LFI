import numpy as np
import torch
import sys
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
import pickle, hickle
import config
from utils import * 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= config.test_param_configs['gpu_id'] # specify which GPU(s) to be used
device = torch.device("cuda:0")
dtype = torch.float32
torch.manual_seed(42)
np.random.seed(42)

def simulate_p_value(n_tr, n_ev, n_te, Model,
                    repeats,
                    test_soft=True, 
                    test_hard=True,
                    plot_hist_path=None):
    p_soft_list = np.zeros(repeats)
    p_hard_list = np.zeros(repeats)
    # run many times
    for j in range(repeats):
        X_test = dataset_P[ np.random.choice(n_tr, n_te, replace=False) ]
        Y_test = dataset_Q[ np.random.choice(n_tr, n_te, replace=False) ]
        X_eval = dataset_P[ n_tr + np.random.choice(dataset_P.shape[0]-n_tr, n_ev, replace=False) ]
        Y_eval = dataset_Q[ n_tr + np.random.choice(dataset_Q.shape[0]-n_tr, n_ev, replace=False) ]
        # Compute the test statistic on the test data
        X_scores = Model.compute_scores(X_eval) # (n_ev,)
        Y_scores = Model.compute_scores(Y_eval) # (n_ev,)
        
        if test_soft:
            p_soft_list[j] = get_pval_from_evaluated_scores(X_scores, Y_scores, 
                                                norm_or_binom=True, thres=None, verbose = False)
        if test_hard:
            # calibrate or use thres=0.5
            X_test_scores = X_scores.cpu(); Y_test_scores = Y_scores.cpu()
            thres, _, _ = get_thres_from_evaluated_scores(X_test_scores, Y_test_scores)
            print("thres = %.4f"%thres)
            p_hard_list[j] = get_pval_from_evaluated_scores(X_scores, Y_scores, 
                                                norm_or_binom=False, thres=thres, verbose = False)
        # plot
        if plot_hist_path != None:
            print("plotting histogram to %s"%plot_hist_path)
            if not os.path.exists(plot_hist_path):
                os.makedirs(plot_hist_path)
            plot_hist(X_scores, Y_scores, plot_hist_path+'/hist.png', title='%f'%p_soft_list[j])

        del X_test, Y_test, X_eval, Y_eval, X_scores, Y_scores
        gc.collect()
        torch.cuda.empty_cache()
    return p_soft_list, p_hard_list

if __name__ == "__main__":
    dataset = np.load(config.resource_configs['Higgs_path'])
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)
    dataset_P = torch.from_numpy(dataset_P).to(dtype=dtype, device=device)
    dataset_Q = torch.from_numpy(dataset_Q).to(dtype=dtype, device=device)
    # load params
    n_tr_list = config.test_param_configs['n_tr_list']
    n_ev = config.test_param_configs['n_ev']
    n_te = config.test_param_configs['n_te']
    num_models = config.test_param_configs['num_models']
    num_repeats = config.test_param_configs['num_repeats']
    pval_path = config.expr_configs['pval_mat_path']
    chekpoints_path = config.expr_configs['checkpoints_path']
    test_soft = config.test_param_configs['test_soft']
    test_hard = config.test_param_configs['test_hard']
    # run tests
    Model = Classifier()
    n_sys_input = int(sys.argv[1])
    method = sys.argv[2]
    for n_tr in [n_sys_input]:
        print("------------------- n_tr = %d -------------------"%n_tr)
        p_soft_mat = np.zeros((num_models,num_repeats)) # p_soft_mat[i,j] = model i and test j
        p_hard_mat = np.zeros((num_models,num_repeats))
        for i in range(num_models):
            n_tr_label = n_tr+i
            path = chekpoints_path+'/n_tr=%d_'%n_tr_label + method
            # Model.load_ckeckpoint(path)
            Model = torch.load(path+'/Model.pt')
            p_soft_mat[i,:], p_hard_mat[i,:] = simulate_p_value(n_tr_label, n_ev, min(n_te,n_tr),
                                                                Model,
                                                                num_repeats,
                                                                test_soft, 
                                                                test_hard,
                                                                plot_hist_path=config.expr_configs['plot_path']+'/n_tr=%d'%n_tr_label)
        if test_soft:
            np.save(pval_path+'/n_tr=%d_soft.npy'%n_tr, p_soft_mat)
            print("p_soft_mean_T = ", np.mean(p_soft_mat))
            print("p_soft_std_T = ", np.std(p_soft_mat))
        if test_hard:
            np.save(pval_path+'/n_tr=%d_hard.npy'%n_tr, p_hard_mat)
            print("p_hard_mean_T = ", np.mean(p_hard_mat))
            print("p_hard_std_T = ", np.std(p_hard_mat))
        gc.collect()
        torch.cuda.empty_cache()
        clear_output(wait=True)
        np.save(pval_path+'/n_tr=%d_soft.npy'%n_tr, p_soft_mat)
        np.save(pval_path+'/n_tr=%d_hard.npy'%n_tr, p_hard_mat)
    
