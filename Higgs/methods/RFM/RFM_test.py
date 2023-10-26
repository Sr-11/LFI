import numpy as np
import torch
import os, sys
import gc
from IPython.display import clear_output
import config as config
from RFM_utils import * 
import RFM_kernels as RFM_kernels
from torch.linalg import solve
import inspect
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:0")
dtype = torch.float32
torch.manual_seed(42)
np.random.seed(42)
torch.set_grad_enabled(False)
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
        X_ev = dataset_P[np.random.choice(n_tr, n_ev, replace=False)]
        Y_ev = dataset_Q[np.random.choice(n_tr, n_ev, replace=False)]
        X_cal = dataset_P[idx]
        Y_cal = dataset_Q[idy]
        #####
        labels =  torch.cat([torch.zeros(n_ev), torch.ones(n_ev)]).to(device)
        labels = labels.reshape(2*n_ev, 1)
        tester = Tester(torch.cat([X_ev, Y_ev]), labels, M)
        # Compute the test statistic on the test data
        X_scores = tester.compute_scores(X_cal) # (n_cal,)
        Y_scores = tester.compute_scores(Y_cal) # (n_cal,)
        if test_soft:
            p_soft_list[j] = get_pval_from_evaluated_scores(X_scores, Y_scores, 
                                                norm_or_binom=True, thres=None, verbose = False)
        del X_ev, Y_ev, X_cal, Y_cal, X_scores, Y_scores
        gc.collect(); torch.cuda.empty_cache()
    if test_soft:
        np.save(os.path.join(folder_path,'pval_orig.npy'), p_soft_list)
    if test_hard:
        np.save(os.path.join(folder_path,'pval_t_opt.npy'), p_hard_list)
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
    if 'n_tr_list' in kwargs:
        ns = json.loads(kwargs['n_tr_list'])
    num_models = config.test_param_configs['num_models']
    num_repeats = config.test_param_configs['num_repeat']

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
            n_ev_ = min(n_ev, n_tr)
            n_tr_label = n_tr
            folder_path = os.path.join(chekpoints_path, 'n_tr=%d#%d'%(n_tr_label, i))
            p_soft_mat[i,:], p_hard_mat[i,:] = simulate_p_value(n_tr, n_cal, n_ev_,
                                                                folder_path=folder_path, 
                                                                repeats=num_repeats,
                                                                test_soft=test_soft, test_hard=test_hard)
        gc.collect()
        torch.cuda.empty_cache()
        clear_output(wait=True)











