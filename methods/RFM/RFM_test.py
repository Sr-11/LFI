import numpy as np
import torch
import os
import pandas as pd
import gc
from IPython.display import clear_output
import pickle, hickle
import config as config
from RFM_utils import * 
import RFM_kernels as RFM_kernels
from torch.linalg import solve
import inspect


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= config.test_param_configs['gpu'] # specify which GPU(s) to be used
device = torch.device("cuda:0")
dtype = torch.float32
torch.manual_seed(42)
np.random.seed(42)
L = 10

class Tester():
    def __init__(self, X_train, y_train, M):
        self.X_train = X_train
        self.y_train = y_train
        self.M = M
    def compute_scores(self, X_in, reg=1e-3): # Z.shape = (m,d)
        K_train = RFM_kernels.laplacian_M(self.X_train, self.X_train, L, self.M)
        sol = solve(K_train + reg * torch.eye(len(K_train), device = device), self.y_train).T # Find the inverse matrix, alpha in the paper
        K_test = RFM_kernels.laplacian_M(self.X_train, X_in, L, self.M)
        preds = (sol @ K_test).T
        return preds
    def compute_test_statistics(self, Zs):# Zs.shape = (num,m,d)
        num = Zs.shape[0]
        stats = torch.zeros(num)
        for i in range(num):
             stats = self.compute_test_statistic(Zs[i])
        return stats

def simulate_p_value(n_tr, n_cal, n_ev,
                    folder_path='', epoch=None,
                    method='RFM', 
                    repeats=10,
                    test_soft=True, test_hard=True):
    M = load_model(folder_path, epoch=epoch)
    # M = M.to(device)
    p_soft_list = np.zeros(len(repeats))
    p_hard_list = np.zeros(len(repeats))
    #####################
    for j in repeats:
        idx = n_tr + np.random.choice(dataset_P.shape[0]-n_tr, n_cal, replace=False) 
        idy = n_tr + np.random.choice(dataset_Q.shape[0]-n_tr, n_cal, replace=False) 
        X_test = dataset_P[np.random.choice(n_tr, n_ev, replace=False)]
        Y_test = dataset_Q[np.random.choice(n_tr, n_ev, replace=False)]
        X_eval = dataset_P[idx]
        Y_eval = dataset_Q[idy]
        #####
        labels =  torch.cat([torch.zeros(n_ev), torch.ones(n_ev)]).to(device)
        labels = labels.reshape(2*n_ev, 1)
        tester = Tester(torch.cat([X_test, Y_test]), labels, M)
        # Compute the test statistic on the test data
        X_scores = tester.compute_scores(X_eval) # (n_cal,)
        Y_scores = tester.compute_scores(Y_eval) # (n_cal,)
        if test_soft:
            p_soft_list[j] = get_pval_from_evaluated_scores(X_scores, Y_scores, 
                                                norm_or_binom=True, thres=None, verbose = False)
            gc.collect()
            torch.cuda.empty_cache()
            # print(os.path.join(folder_path,'pval_orig.npy'))
            
        if test_hard:
            # calibrate / use thres=0.5
            # X_test_scores = tester.compute_scores(X_test) # (n_cal,)
            # Y_test_scores = tester.compute_scores(Y_test) # (n_cal,)
            X_test_scores = X_scores; Y_test_scores = Y_scores
            X_test_scores = X_test_scores.cpu()
            Y_test_scores = Y_test_scores.cpu()
            thres, _, _ = get_thres_from_evaluated_scores(X_test_scores, Y_test_scores)
            print("thres = %.4f"%thres)
            # thres = 0.5
            p_hard_list[j] = get_pval_from_evaluated_scores(X_scores, Y_scores, 
                                                norm_or_binom=False, thres=thres, verbose = False)

        del X_test, Y_test, X_eval, Y_eval, X_scores, Y_scores
        gc.collect()
        torch.cuda.empty_cache()
    if test_soft:
        np.save(os.path.join(folder_path,'pval_orig.npy'), p_soft_list)
    if test_hard:
        np.save(os.path.join(folder_path,'pval_t_opt.npy'), p_hard_list)
    return p_soft_list, p_hard_list

def simulate_error(n_tr, n_cal, n_ev,
                    folder_path='', epoch=None,
                    method='RFM', 
                    repeats=10,
                    test_soft=True, test_hard=False):
    M = load_model(folder_path, epoch=epoch)
    # M = M.to(device)
    error_soft_list = np.zeros(repeats)
    error_hard_list = np.zeros(repeats)
    #####################
    for j in range(repeats):
        idx = n_tr + np.random.choice(dataset_P.shape[0]-n_tr, n_cal, replace=False) 
        idy = n_tr + np.random.choice(dataset_Q.shape[0]-n_tr, n_cal, replace=False) 
        X_test = dataset_P[np.random.choice(n_tr, n_ev, replace=False)]
        Y_test = dataset_Q[np.random.choice(n_tr, n_ev, replace=False)]
        X_eval = dataset_P[idx]
        Y_eval = dataset_Q[idy]
        #####
        labels =  torch.cat([torch.zeros(n_ev), torch.ones(n_ev)]).to(device)
        labels = labels.reshape(2*n_ev, 1)
        tester = Tester(torch.cat([X_test, Y_test]), labels, M)
        # Compute the test statistic on the test data
        X_scores = tester.compute_scores(X_eval) # (n_cal,)
        Y_scores = tester.compute_scores(Y_eval) # (n_cal,)
        if test_soft:
            error_soft_list[j] = get_error_from_evaluated_scores(X_scores, Y_scores, 
                                                m=1100, pi=1/11, gamma=0, 
                                                norm_or_binom=True, thres=None, verbose = False)
        if test_hard:
            # calibrate / use thres=0.5
            # X_test_scores = tester.compute_scores(X_test) # (n_cal,)
            # Y_test_scores = tester.compute_scores(Y_test) # (n_cal,)
            # X_test_scores = X_scores; Y_test_scores = Y_scores
            # X_test_scores = X_test_scores.cpu()
            # Y_test_scores = Y_test_scores.cpu()
            # thres, _, _ = get_thres_from_evaluated_scores(X_test_scores, Y_test_scores)
            # print("thres = %.4f"%thres)
            # # thres = 0.5
            # error_hard_list[j] = get_pval_from_evaluated_scores(X_scores, Y_scores, 
            #                                     norm_or_binom=False, thres=thres, verbose = False)
            pass
            
        del X_test, Y_test, X_eval, Y_eval, X_scores, Y_scores
        gc.collect()
        torch.cuda.empty_cache()
    return error_soft_list, error_hard_list


def package_pval_data(pval_path):
    ns = config.test_param_configs['n_tr_list']
    pval_mid_path = config.expr_configs['pval_mat_path']

    num_models = config.test_param_configs['num_models']
    num_repeats = config.test_param_configs['num_repeats']

    pval_dict = {}
    pval_dict['RFM'] = {}
    pval_dict['RFM']['soft'] = {}
    pval_dict['RFM']['hard'] = {}

    for n_tr in ns:
        print("------------------- n_tr = %d -------------------"%n_tr)
        p_soft_mat = np.load(pval_path+'/n_tr=%d_soft.npy'%n_tr)
        p_hard_mat = np.load(pval_path+'/n_tr=%d_hard.npy'%n_tr)
        assert p_soft_mat.shape == p_hard_mat.shape == (num_models, num_repeats)
        pval_dict['RFM']['soft'][n_tr] = {}
        pval_dict['RFM']['hard'][n_tr] = {}
        pval_dict['RFM']['soft'][n_tr]['data'] = p_soft_mat
        pval_dict['RFM']['hard'][n_tr]['data'] = p_hard_mat
        pval_dict['RFM']['soft'][n_tr]['mean'] = np.mean(p_soft_mat)
        pval_dict['RFM']['hard'][n_tr]['mean'] = np.mean(p_hard_mat)
        pval_dict['RFM']['soft'][n_tr]['std'] = np.std(p_soft_mat)
        pval_dict['RFM']['hard'][n_tr]['std'] = np.std(p_hard_mat)

    with open(pval_path+'/pval_dict.pkl', 'wb') as f:
        pickle.dump(pval_dict, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    dataset = np.load(config.resource_configs['Higgs_path'])
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)
    dataset_P = torch.from_numpy(dataset_P).to(dtype=dtype, device=device)
    dataset_Q = torch.from_numpy(dataset_Q).to(dtype=dtype, device=device)

    ns = config.test_param_configs['n_tr_list']
    num_models = config.test_param_configs['num_models']
    num_repeats = config.test_param_configs['num_repeats']

    n_cal = config.test_param_configs['n_cal']
    n_ev = config.test_param_configs['n_ev']

    current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))) 
    pval_path = os.path.join(current_dir, 'checkpoints')
    chekpoints_path = os.path.join(current_dir, 'checkpoints')

    test_soft = config.test_param_configs['test_soft']
    test_hard = config.test_param_configs['test_hard']

    for n_tr in ns:
        print("------------------- n_tr = %d -------------------"%n_tr)
        p_soft_mat = np.zeros([len(num_models),len(num_repeats)]) # p_soft_mat[i,j] = model i and test j
        p_hard_mat = np.zeros([len(num_models),len(num_repeats)])
        for i in trange(len(num_models)):
            n_tr_label = n_tr
            folder_path = chekpoints_path+'/n_tr=%d#%d'%(n_tr_label, i)
            p_soft_mat[i,:], p_hard_mat[i,:] = simulate_p_value(n_tr, n_cal, n_ev,
                                                                folder_path=folder_path, 
                                                                method='RFM', 
                                                                repeats=num_repeats,
                                                                test_soft=test_soft, test_hard=test_hard)
        if test_soft:
            np.save(pval_path+'/n_tr=%d_soft.npy'%n_tr, p_soft_mat[i,:])
        if test_hard:
            np.save(pval_path+'/n_tr=%d_hard.npy'%n_tr, p_hard_mat[i,:])
        gc.collect()
        torch.cuda.empty_cache()
        clear_output(wait=True)
