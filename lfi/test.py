import numpy as np
import torch
import sys
# from matplotlib import pyplot as plt
# import torch.nn as nn
# import time
# from numba import cuda
from tqdm import tqdm, trange
import os
# import pyroc
# import pandas as pd
import gc
from IPython.display import clear_output
# import pickle, hickle
from .utils import *
import importlib.util

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:0")
dtype = torch.float32
torch.manual_seed(42)
np.random.seed(42)

def simulate_p_value(dataset_P, dataset_Q, 
                    n_tr, n_ev, n_te, kernel, repeats,
                    test_soft, test_hard,
                    plot_hist_path):
    p_soft_list = np.zeros(repeats)
    p_hard_list = np.zeros(repeats)
    # run many times
    for j in tqdm(range(repeats), desc='progress in n_tr=%d'%n_tr):
        X_test = dataset_P[ np.random.choice(n_tr, n_te, replace=False) ]
        Y_test = dataset_Q[ np.random.choice(n_tr, n_te, replace=False) ]
        X_eval = dataset_P[ n_tr + np.random.choice(dataset_P.shape[0]-n_tr, n_ev, replace=False) ]
        Y_eval = dataset_Q[ n_tr + np.random.choice(dataset_Q.shape[0]-n_tr, n_ev, replace=False) ]
        # Compute the test statistic on the test data
        X_scores = kernel.compute_scores(X_test, Y_test, X_eval) # (n_ev,)
        Y_scores = kernel.compute_scores(X_test, Y_test, Y_eval) # (n_ev,)
        # plot
        if plot_hist_path != None:
            print("plotting histogram to %s"%(plot_hist_path+'/hist.png'))
            plot_hist(X_scores, Y_scores, plot_hist_path+'/hist.png', title='')
        # compute p-value
        if test_soft:
            p_soft_list[j] = get_pval_from_evaluated_scores(X_scores, Y_scores, thres=None, verbose = False)
            print("p_soft = %.4f"%p_soft_list[j])
        if test_hard:
            # calibrate or use thres=0.5
            X_test_scores = X_scores.cpu(); Y_test_scores = Y_scores.cpu()
            thres, _, _ = get_thres_from_evaluated_scores(X_test_scores, Y_test_scores)
            # print("thres = %.4f"%thres)
            p_hard_list[j] = get_pval_from_evaluated_scores(X_scores, Y_scores, 
                                                thres=thres, verbose = False)
            print("p_hard = %.4f"%p_hard_list[j])
        del X_test, Y_test, X_eval, Y_eval, X_scores, Y_scores
        gc.collect()
        torch.cuda.empty_cache()
    return p_soft_list, p_hard_list

def simulate_error(n_tr, n_ev, n_te, kernel,
                    repeats,
                    test_soft=True, 
                    test_hard=True,
                    plot_hist_path=None):
    type_1_error_list = np.zeros(repeats)
    type_2_error_list = np.zeros(repeats)
    # run many times
    for j in range(repeats):
        X_test = dataset_P[ np.random.choice(n_tr, n_te, replace=False) ]
        Y_test = dataset_Q[ np.random.choice(n_tr, n_te, replace=False) ]
        X_eval = dataset_P[ n_tr + np.random.choice(dataset_P.shape[0]-n_tr, n_ev, replace=False) ]
        Y_eval = dataset_Q[ n_tr + np.random.choice(dataset_Q.shape[0]-n_tr, n_ev, replace=False) ]
        # Compute the test statistic on the test data
        X_scores = kernel.compute_scores(X_test, Y_test, X_eval) # (n_ev,)
        Y_scores = kernel.compute_scores(X_test, Y_test, Y_eval) # (n_ev,)
        # Compute test
        gamma = kernel.compute_gamma(X_test, Y_test, pi)
        type_1_error, type_2_error = get_error_from_evaluated_scores(X_score, Y_score, pi, gamma=gamma)
        type_1_error_list[j] = type_1_error
        type_2_error_list[j] = type_2_error
        # Delete
        del X_test, Y_test, X_eval, Y_eval, X_scores, Y_scores
        gc.collect(); torch.cuda.empty_cache()
    return type_1_error_list, type_2_error_list

def main_pval(config_path):
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
    # load data
    os.environ["CUDA_VISIBLE_DEVICES"]= config.test_param_configs['gpu_id'] 
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
    test_soft = config.test_param_configs['test_soft']
    test_hard = config.test_param_configs['test_hard']
    pval_path = config.expr_configs['pval_mat_path']
    chekpoints_path = config.expr_configs['checkpoints_path']
    plot_path = config.expr_configs['plot_path']
    # run tests
    for n_tr in n_tr_list:
        p_soft_mat = np.zeros((num_models,num_repeats)) # p_soft_mat[i,j] = model i and test j
        p_hard_mat = np.zeros((num_models,num_repeats))
        for i in range(num_models):
            n_tr_label = n_tr+i
            print("\n------------------- TEST n_tr = %d -------------------"%n_tr_label)
            path = chekpoints_path+'/n_tr=%d'%n_tr_label
            kernel = torch.load(path+'/kernel.pt')
            plot_hist_path = plot_path+'/n_tr=%d'%n_tr_label
            p_soft_mat[i,:], p_hard_mat[i,:] = simulate_p_value(dataset_P, dataset_Q,
                                                                n_tr_label, n_ev, min(n_te,n_tr),
                                                                kernel, num_repeats,
                                                                test_soft, test_hard,
                                                                plot_hist_path)
        if test_soft:
            np.save(pval_path+'/n_tr=%d_soft.npy'%n_tr, p_soft_mat)
            print("p_soft_mat saved to ", pval_path+'/n_tr=%d_soft.npy'%n_tr)
            print("p_soft_mean = ", np.mean(p_soft_mat))
            print("p_soft_std = ", np.std(p_soft_mat))
        if test_hard:
            np.save(pval_path+'/n_tr=%d_hard.npy'%n_tr, p_hard_mat)
            print("p_hard_mat saved to ", pval_path+'/n_tr=%d_hard.npy'%n_tr)
            print("p_hard_mean = ", np.mean(p_hard_mat))
            print("p_hard_std = ", np.std(p_hard_mat))
        gc.collect()
        torch.cuda.empty_cache()
        clear_output(wait=True)
        np.save(pval_path+'/n_tr=%d_soft.npy'%n_tr, p_soft_mat)
        np.save(pval_path+'/n_tr=%d_hard.npy'%n_tr, p_hard_mat)
    
if __name__ == "__main__":
    config_path = sys.argv[1]
    main_pval(config_path)



# def save_pval_data(pval_path):
#     ns = config.test_param_configs['n_tr_list']

#     num_models = config.test_param_configs['num_models']
#     num_repeats = config.test_param_configs['num_repeats']

#     pval_dict = {}
#     pval_dict['RFM'] = {}
#     pval_dict['RFM']['soft'] = {}
#     pval_dict['RFM']['hard'] = {}

#     for n_tr in ns:
#         print("------------------- n_tr = %d -------------------"%n_tr)
#         p_soft_mat = np.load(pval_path+'/n_tr=%d_soft.npy'%n_tr)
#         p_hard_mat = np.load(pval_path+'/n_tr=%d_hard.npy'%n_tr)
#         assert p_soft_mat.shape == p_hard_mat.shape == (num_models, num_repeats)
#         pval_dict['RFM']['soft'][n_tr] = {}
#         pval_dict['RFM']['hard'][n_tr] = {}
#         pval_dict['RFM']['soft'][n_tr]['data'] = p_soft_mat
#         pval_dict['RFM']['hard'][n_tr]['data'] = p_hard_mat
#         pval_dict['RFM']['soft'][n_tr]['mean'] = np.mean(p_soft_mat)
#         pval_dict['RFM']['hard'][n_tr]['mean'] = np.mean(p_hard_mat)
#         pval_dict['RFM']['soft'][n_tr]['std'] = np.std(p_soft_mat)
#         pval_dict['RFM']['hard'][n_tr]['std'] = np.std(p_hard_mat)

#     with open(pval_path+'/pval_dict.pkl', 'wb') as f:
#         pickle.dump(pval_dict, f, pickle.HIGHEST_PROTOCOL)