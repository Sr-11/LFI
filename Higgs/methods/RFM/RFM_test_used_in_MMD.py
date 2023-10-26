import numpy as np
import torch
import sys, os, gc
from tqdm import tqdm, trange
import pandas as pd
from IPython.display import clear_output
import config
import RFM_kernels
import rfm
from RFM_utils import *
import inspect

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
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
        self.P_train = X_train[(y_train==0).flatten(),:]
        self.Q_train = X_train[(y_train==1).flatten(),:]
    def compute_scores(self, X_in, reg=1e-3): # Z.shape = (m,d)
        t1 = RFM_kernels.laplacian_M(X_in, self.P_train, L, self.M)
        t2 = RFM_kernels.laplacian_M(X_in, self.Q_train, L, self.M)
        scores = torch.mean(t2, dim=1) - torch.mean(t1, dim=1)
        return scores

def simulate_p_value(n_tr, n_cal, n_ev,
                    folder_path='', epoch=None,
                    method='RFM', 
                    repeats=10,
                    test_soft=True, test_hard=True, r=0):
    M = load_model(folder_path, epoch=epoch)
    # M = M.to(device)
    p_soft_list = np.zeros(len(repeats))
    p_hard_list = np.zeros(len(repeats))
    #####################
    for j in tqdm(repeats, desc='n_tr=%d, n_cal=%d, n_ev=%d, model_num=%d'%(n_tr, n_cal, n_ev, r)):
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
    return p_soft_list, p_hard_list

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]= config.test_param_configs['gpu'] # specify which GPU(s) to be used
    kwargs = dict([arg.split('=') for arg in sys.argv[1:]])
    if 'gpu' in kwargs:
        os.environ["CUDA_VISIBLE_DEVICES"] = kwargs['gpu']

    dataset = np.load(config.resource_configs['Higgs_path'])
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)
    dataset_P = torch.from_numpy(dataset_P).to(dtype=dtype, device=device)
    dataset_Q = torch.from_numpy(dataset_Q).to(dtype=dtype, device=device)

    ns = config.test_param_configs['n_tr_list']
    num_models = config.test_param_configs['num_models']
    num_repeat = config.test_param_configs['num_repeat']

    n_cal = config.test_param_configs['n_cal']
    n_ev = config.test_param_configs['n_ev']

    test_soft = config.test_param_configs['test_soft']
    test_hard = config.test_param_configs['test_hard']

    current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))) 
    pval_path = os.path.join(current_dir, 'checkpoints')
    chekpoints_path = os.path.join(current_dir, 'checkpoints')

    for n_tr in ns:
        print("------------------- n_tr = %d -------------------"%n_tr)
        for i in num_models:
            folder_path = os.path.join(chekpoints_path, 'n_tr=%d#%d'%(n_tr, i))
            p_soft_list, p_hard_list = simulate_p_value(n_tr, n_cal, min(n_ev,n_tr),
                                                                folder_path=folder_path, 
                                                                method='RFM', 
                                                                repeats=num_repeat,
                                                                test_soft=test_soft, test_hard=test_hard, r=i)
            if test_soft:
                print("pval_as_mmd = %.4f"%(np.mean(p_soft_list)))
                np.save(os.path.join(folder_path,'pval_as_mmd.npy'), p_soft_list)
            if test_hard:
                print("pval_as_mmd_t_opt = %.4f"%(np.mean(p_hard_list)))
                np.save(os.path.join(folder_path,'pval_as_mmd_t_opt.npy'), p_hard_list)
        gc.collect()
        torch.cuda.empty_cache()
        clear_output(wait=True)
