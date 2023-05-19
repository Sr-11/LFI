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
import pandas as pd

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:0")
dtype = torch.float32
torch.manual_seed(42)
np.random.seed(42)

def simulate_p_value(dataset_P, dataset_Q, 
                    n_tr, n_ev, n_te, kernel, repeats,
                    test_soft, test_hard, force_thres,
                    plot_hist_path):
    p_soft_list = np.zeros(repeats)
    p_hard_list = np.zeros(repeats)
    p_force_thres_list = np.zeros(repeats)
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
            X_test_scores = X_scores.cpu(); Y_test_scores = Y_scores.cpu()
            thres, _, _ = get_thres_from_evaluated_scores(X_test_scores, Y_test_scores)
            # print("thres = %.4f"%thres)
            p_hard_list[j] = get_pval_from_evaluated_scores(X_scores, Y_scores, 
                                                thres=thres, verbose = False)
            print("p_hard = %.4f"%p_hard_list[j])
        if force_thres != None:
            p_force_thres_list[j] = get_pval_from_evaluated_scores(X_scores, Y_scores, 
                                            thres=force_thres, verbose = False)
        del X_test, Y_test, X_eval, Y_eval, X_scores, Y_scores
        gc.collect()
        torch.cuda.empty_cache()
    return p_soft_list, p_hard_list, p_force_thres_list

def simulate_error(dataset_P, dataset_Q,
                   n_tr, n_ev, n_te, 
                   kernel, repeats,
                   pi, m,
                   batch_size,
                   plot_hist_path=None,
                   callback=None):
    m_num = m.shape[0]
    type_1_error_list = np.zeros([m_num, repeats])
    type_2_error_list = np.zeros([m_num, repeats])
    # run many times
    for j in tqdm(range(repeats), desc='progress of repeating in n_tr=%d'%n_tr):
        idx = np.random.choice(dataset_P.shape[0]-n_tr, n_te+n_ev, replace=False)+n_tr
        idy = np.random.choice(dataset_Q.shape[0]-n_tr, n_te+n_ev, replace=False)+n_tr
        # X_test = dataset_P[ np.random.choice(n_tr, n_te, replace=False) ]
        # X_eval = dataset_P[ n_tr + np.random.choice(dataset_P.shape[0]-n_tr, n_ev, replace=False) ]
        # Y_test = dataset_Q[ np.random.choice(n_tr, n_te, replace=False) ]
        # Y_eval = dataset_Q[ n_tr + np.random.choice(dataset_Q.shape[0]-n_tr, n_ev, replace=False) ]
        X_test = dataset_P[ idx[:n_te] ]
        X_eval = dataset_P[ idx[n_te:] ]
        Y_test = dataset_Q[ idy[:n_te] ]
        Y_eval = dataset_Q[ idy[n_te:] ]

        # Compute the test statistic on the test data
        X_scores = kernel.compute_scores(X_test, Y_test, X_eval, batch_size=batch_size) # (n_ev,)
        Y_scores = kernel.compute_scores(X_test, Y_test, Y_eval, batch_size=batch_size) # (n_ev,)
        gamma = kernel.compute_gamma(X_test, Y_test, pi)
        # gamma = (1-pi/2)*torch.mean(X_scores) + pi/2*torch.mean(Y_scores)
        type_1_error, type_2_error = get_error_from_evaluated_scores(X_scores, Y_scores, pi, gamma, m)
        type_1_error_list[:, j] = type_1_error
        type_2_error_list[:, j] = type_2_error
        # plot hist
        if plot_hist_path != None:
            title = 'n_ev=%d, n_te=%d, n_tr=%d, repeat=%d'%(n_ev, n_te, n_tr, j)
            print("plotting histogram to %s"%(plot_hist_path))
            print(title)
            plot_hist(X_scores, Y_scores, plot_hist_path, title, pi=pi, gamma=gamma)
        # Delete
        del X_scores, Y_scores, gamma; gc.collect(); torch.cuda.empty_cache()
        # callback
        if callback != None:
            callback()
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
    force_thres = config.test_param_configs['force_thres']
    chekpoints_path = config.expr_configs['checkpoints_path']
    plot_path = config.expr_configs['plot_path']
    # run tests
    for n_tr in n_tr_list:
        p_soft_mat = np.zeros((num_models,num_repeats)) # p_soft_mat[i,j] = model i and test j
        p_hard_mat = np.zeros((num_models,num_repeats))
        p_force_thres_mat = np.zeros((num_models,num_repeats))
        for i in range(num_models):
            n_tr_label = n_tr+i
            print("\n------------------- TEST n_tr = %d -------------------"%n_tr_label)
            path = chekpoints_path+'/n_tr=%d'%n_tr_label
            kernel = torch.load(path+'/kernel.pt')
            plot_hist_path = plot_path+'/n_tr=%d'%n_tr_label
            p_soft_mat[i,:], p_hard_mat[i,:], p_force_thres_mat[i,:] = simulate_p_value(dataset_P, dataset_Q,
                                                                        n_tr_label, n_ev, min(n_te,n_tr),
                                                                        kernel, num_repeats,
                                                                        test_soft, test_hard, force_thres,
                                                                        plot_hist_path)
        if os.path.exists(pval_path) == False:
            os.makedirs(pval_path)
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
        if force_thres != None:
            np.save(pval_path+'/n_tr=%d_force_thres.npy'%n_tr, p_force_thres_mat)
            print("p_force_thres_mat saved to ", pval_path+'/n_tr=%d_force_thres.npy'%n_tr)
            print("p_force_thres_mean = ", np.mean(p_force_thres_mat))
            print("p_force_thres_std = ", np.std(p_force_thres_mat))

        gc.collect()
        torch.cuda.empty_cache()
        clear_output(wait=True)
        # np.save(pval_path+'/n_tr=%d_soft.npy'%n_tr, p_soft_mat)
        # np.save(pval_path+'/n_tr=%d_hard.npy'%n_tr, p_hard_mat)

def main_error(config_path):
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
    pval_path = config.expr_configs['pval_mat_path']
    force_thres = config.test_param_configs['force_thres']
    chekpoints_path = config.expr_configs['checkpoints_path']
    plot_path = config.expr_configs['plot_path']
    # run tests
    pi = 0.1
    df = pd.DataFrame(columns = ['n_tr', 'n_te', 'id', 'm', 'pi', 'error'])
    for n_tr in n_tr_list:
        for i in range(num_models):
            n_tr_label = n_tr+i
            print("\n------------------- TEST n_tr = %d -------------------"%n_tr_label)
            path = chekpoints_path+'/n_tr=%d'%n_tr_label
            kernel = torch.load(path+'/kernel.pt')
            plot_hist_path = plot_path+'/n_tr=%d'%n_tr_label
            error = lfi.test.simulate_error(dataset_P, dataset_Q,
                                            1300000, n_ev, n_te, 
                                            kernel, repeats,
                                            pi, m_list,
                                            batch_size,
                                            plot_hist_path=None)
        if os.path.exists(pval_path) == False:
            os.makedirs(pval_path)
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
        if force_thres != None:
            np.save(pval_path+'/n_tr=%d_force_thres.npy'%n_tr, p_force_thres_mat)
            print("p_force_thres_mat saved to ", pval_path+'/n_tr=%d_force_thres.npy'%n_tr)
            print("p_force_thres_mean = ", np.mean(p_force_thres_mat))
            print("p_force_thres_std = ", np.std(p_force_thres_mat))

        gc.collect()
        torch.cuda.empty_cache()
        clear_output(wait=True)
        # np.save(pval_path+'/n_tr=%d_soft.npy'%n_tr, p_soft_mat)
        # np.save(pval_path+'/n_tr=%d_hard.npy'%n_tr, p_hard_mat)    

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
