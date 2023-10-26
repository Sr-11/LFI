import numpy as np
import torch
from utils import *
import os, sys
import gc
from IPython.display import clear_output
import inspect
import config
import json
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:0")
dtype = torch.float32
torch.manual_seed(42)
np.random.seed(42)
class DN(torch.nn.Module):
    def __init__(self, H=300, out=100):
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
    def __init__(self, H=300, out=100):
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

def get_p(n_train, num_model, path, method, force_thres = None):
    model = DN(300, 100).cuda()
    another_model = another_DN(300, 100).cuda()
    model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst = load_model(model, another_model, path)
    gc.collect()
    torch.cuda.empty_cache()
    #####################
    p_soft_list = np.zeros(10)
    p_hard_list = np.zeros(10)
    n = n_train
    for i in range(10):
        gc.collect(); torch.cuda.empty_cache()
        n_ev = min(n,32768)
        n_cal = 32768
        idx = np.random.choice(dataset_P.shape[0]-n_train, n_cal+n_ev, replace=False) + n_train
        idy = np.random.choice(dataset_Q.shape[0]-n_train, n_cal+n_ev, replace=False) + n_train
        X_ev = dataset_P[np.random.choice(n_train, n_ev, replace=False)]
        Y_ev = dataset_Q[np.random.choice(n_train, n_ev, replace=False)]
        X_cal = dataset_P[idx][n_ev:n_ev+n_cal]
        Y_cal = dataset_Q[idy][n_ev:n_ev+n_cal]
        X_ev_eval = X_ev
        Y_ev_eval = Y_ev
        print('n_ev=%d, n_cal=%d, n_tr=%d, num_model=%d, repeat=%d'%(n_ev, n_cal, n, num_model, i))
        p_soft = get_pval_at_once(X_ev, Y_ev, X_ev_eval, Y_ev_eval, X_cal, Y_cal,
                        model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,
                        batch_size = 10000,
                        norm_or_binom=True)
        p_hard = get_pval_at_once(X_ev, Y_ev, X_ev_eval, Y_ev_eval, X_cal, Y_cal,
                        model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,
                        batch_size = 10000,
                        norm_or_binom=False)
        p_soft_list[i] = p_soft
        p_hard_list[i] = p_hard
        clear_output(wait=True)
    return p_soft_list, p_hard_list

if __name__ == '__main__':
    kwargs = dict([arg.split('=') for arg in sys.argv[1:]])
    current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))) 

    os.environ["CUDA_VISIBLE_DEVICES"]= config.train_param_configs['gpu']
    if 'gpu' in kwargs:
        os.environ["CUDA_VISIBLE_DEVICES"]= kwargs['gpu']
    kwargs = dict([arg.split('=') for arg in sys.argv[1:]])

    dataset = np.load(os.path.join(current_dir, '..', '..', 'datasets', 'HIGGS.npy'))
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)
    dataset_P = MatConvert(dataset_P, device=device, dtype=dtype)
    dataset_Q = MatConvert(dataset_Q, device=device, dtype=dtype)


    ns = config.test_param_configs['n_tr_list']

    if 'n_tr_list' in kwargs:
        ns = json.loads(kwargs['n_tr_list'])

    models = 10
    cur_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
    for n in ns:
        print('n_tr=%d'%n)
        for i in tqdm(range(models), desc='n_tr=%d'%n):
            print('model %d'%i)
            ckpt_dir = os.path.join(cur_dir, 'checkpoints', 'n_tr=%d#%d'%(ns[0], i))
            p_soft_list, p_hard_list = get_p(n,i, ckpt_dir, 'Mix', None)
            np.save(os.path.join(ckpt_dir, 'pval_orig.npy'), p_soft_list)
            np.save(os.path.join(ckpt_dir, 'pval_t_opt.npy'), p_hard_list)
            print('pval_orig: ', p_soft_list)
            print('pval_t_opt: ', p_hard_list)