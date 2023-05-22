import numpy as np
import torch
import sys
from utils import *
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
import gc
import sys
# sys.path.append(sys.path[0]+"/..")
# import global_config

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7" 
torch.backends.cudnn.deterministic = True
dtype = torch.float
device = torch.device("cuda:0")
torch.manual_seed(42)
np.random.seed(42)

# define network
H = 300
out = 100
x_in = 28
L = 1
class DN(torch.nn.Module):
    def __init__(self):
        super(DN, self).__init__()
        self.restored = False
        self.model = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
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

# define loss function
def crit(mmd_val, mmd_var, liuetal=True, Sharpe=False):
    if liuetal:
        mmd_std_temp = torch.sqrt(mmd_var+10**(-8)) #this is std
        return torch.div(mmd_val, mmd_std_temp)

# save checkpoint
def save_model(n,model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst,epoch=0):
    path = sys.path[0]+'/checkpoint%d/'%n+str(epoch)+'/'
    try:
        os.makedirs(path) 
    except:
        pass
    torch.save(model.state_dict(), path+'model.pt')
    torch.save(another_model.state_dict(), path+'another_model.pt')
    torch.save(epsilonOPT, path+'epsilonOPT.pt')
    torch.save(sigmaOPT, path+'sigmaOPT.pt')
    torch.save(sigma0OPT, path+'sigma0OPT.pt')
    torch.save(eps, path+'eps.pt')
    torch.save(cst, path+'cst.pt')

# load checkpoint
def load_model(n, epoch=0):
    path = './checkpoint%d/'%n+str(epoch)+'/'
    model = DN().cuda()
    model.load_state_dict(torch.load(path+'model.pt'))
    epsilonOPT = torch.load(path+'epsilonOPT.pt')
    sigmaOPT = torch.load(path+'sigmaOPT.pt')
    sigma0OPT = torch.load(path+'sigma0OPT.pt')
    eps = torch.load(path+'eps.pt')
    cst = torch.load(path+'cst.pt')
    return model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst

# train
def train(n, learning_rate=5e-4, 
            K=10, N=1000, N_epoch=50, 
            print_every=1, batch_size=32, 
            SGD=True, 
            dataset_P=None, dataset_Q=None,
            momentum = 0.9, weight_decay=0.0,):  
    n_backup = n
    try:
        os.mkdir(sys.path[0]+'/checkpoint%d'%n_backup)
    except:
        pass
    #cuda.select_device(0)
    batches = (n-1)//batch_size + 1 # last batch could be empty
    n = batches*batch_size  
    print("------------------------------------------------")
    X = dataset_P[0:n]
    Y = dataset_Q[0:n]
    print(X.shape, Y.shape)
    total_S = [(X[i*batch_size:(i+1)*batch_size], 
                Y[i*batch_size:(i+1)*batch_size]) 
                for i in range(batches)]
    total_S = [MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in total_S]
    # prepare NN
    model = DN().to(device)
    another_model = another_DN().to(device)
    # prepare other parameters
    epsilonOPT = MatConvert(-np.random.rand(1), device, dtype)
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.sqrt(np.random.rand(1)), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.sqrt(np.random.rand(1)), device, dtype)
    sigma0OPT.requires_grad = True
    eps=MatConvert(np.zeros((1,)), device, dtype)
    eps.requires_grad = True
    cst=MatConvert(1*np.ones((1,)), device, dtype) # set to 1 to meet liu etal objective
    cst.requires_grad = False
    # prepare optimizer
    params = list(model.parameters())+list(another_model.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT]+[eps]+[cst]
    if SGD:
        optimizer_u = torch.optim.SGD(params, lr=learning_rate,
                                    momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer_u = torch.optim.Adam(params, lr=learning_rate)
    # validation data
    S1_v = np.concatenate((dataset_P[n + np.random.choice(n, min(n,10000), replace=False)], 
                            dataset_Q[n + np.random.choice(n, min(n,10000), replace=False)]), axis=0)
    S1_v = MatConvert(S1_v, device, dtype)
    J_validations = np.ones([N_epoch])*np.inf
    mmd_val_validations = np.zeros([N_epoch])
    #############################
    #############################
    for t in range(N_epoch):
        print('epoch %d'%t)
        order = np.random.permutation(batches)
        for ind in tqdm(order):
            optimizer_u.zero_grad()
            ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
            sigma = sigmaOPT ** 2
            sigma0_u = sigma0OPT ** 2
            S = total_S[ind]
            modelu_output = model(S) 
            another_output = another_model(S)
            TEMP = MMDu(modelu_output, batch_size, another_output, sigma, sigma0_u, ep, cst, L=L) # could raise error
            mmd_val = -1 * TEMP[0]
            mmd_var = TEMP[1]
            STAT_u = crit(mmd_val, mmd_var) 
            STAT_u.backward(retain_graph=False)
            optimizer_u.step()      
        # validation
        with torch.torch.no_grad():
            modelu_output = model(S1_v)
            another_output = another_model(S1_v)
            TEMP = MMDu(modelu_output, min(n,10000), another_output, sigma, sigma0_u, ep, cst, L=L)
            mmd_value_temp, mmd_var_temp = -TEMP[0], TEMP[1]
            mmd_val_validations[t] = mmd_value_temp.item()
            J_validations[t] = crit(mmd_value_temp, mmd_var_temp).item()
            print('J_validations =', J_validations[t])
            #print(sigmaOPT.item(), sigma0OPT.item(), epsilonOPT.item())
        if t%print_every==0:
            plt.plot(J_validations[:t])
            plt.savefig(sys.path[0]+'/checkpoint%d/J_validations.png'%n_backup)
            plt.clf()
            save_model(n_backup,model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst,epoch=t)
                
        if early_stopping(J_validations, t) and J_validations[t]<-0.001:
            plt.plot(J_validations[:t])
            plt.savefig(sys.path[0]+'/checkpoint%d/J_validations.png'%n_backup)
            plt.clf()
            save_model(n_backup,model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst,epoch=0)
            return model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,J_validations[t]
    return model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,J_validations[t]


if __name__ == "__main__":
    # load data, please use .npy ones (40000), .gz (10999999)(11e6) is too large.
    
    dataset = np.load(sys.path[0]+'/../../datasets/HIGGS.npy')
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)

    n_list = []
    for i in range(10):
        for n in [1300000]:
            n_list.append(n+i)

    print_every = 1
    for n in n_list:
        gc.collect()
        torch.cuda.empty_cache()
        while True:
            print('-------------------')
            print(n)
            model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,J = train(n, 
                N_epoch = 501, # 只load就设成1
                print_every = print_every, 
                batch_size = 1024, 
                learning_rate =2e-3, 
                SGD = True, 
                dataset_P = dataset_P, dataset_Q = dataset_Q, #Mixture
                momentum=0.99, weight_decay=0.000)
            if J<-0.01:
                break