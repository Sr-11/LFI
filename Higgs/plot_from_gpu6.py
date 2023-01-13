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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"  # specify which GPU(s) to be used

H = 300
out= 100
L = 1
relu = True
if not relu:
    class DN(torch.nn.Module):
        def __init__(self):
            super(DN, self).__init__()
            self.restored = False
            self.model = torch.nn.Sequential(
                torch.nn.Linear(28, H, bias=True),
                torch.nn.Softplus(),
                torch.nn.Linear(H, H, bias=True),
                torch.nn.Softplus(),
                torch.nn.Linear(H, H, bias=True),
                torch.nn.Softplus(),
                torch.nn.Linear(H, H, bias=True),
                torch.nn.Softplus(),
                torch.nn.Linear(H, H, bias=True),
                torch.nn.Softplus(),
                torch.nn.Linear(H, out, bias=True),
            )
        def forward(self, input):
            output = self.model(input)
            return output
if relu:
    class DN(torch.nn.Module):
        def __init__(self):
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


def crit(mmd_val, mmd_var, liuetal=True, Sharpe=False):
    """compute the criterion, liu or Sharpe"""
    ######IMPORTANT: if we want to maximize, need to multiply by -1######
    #return mmd_val + mmd_var
    if liuetal:
        mmd_std_temp = torch.sqrt(mmd_var+10**(-8)) #this is std
        return torch.div(mmd_val, mmd_std_temp)
    # elif Sharpe:
    #     return mmd_val - 2.0 * mmd_var

# calculate the MMD for m!=n
def mmdG(X, Y, model_u, n, sigma, sigma0_u, device, dtype, ep, L=L):
    S = np.concatenate((X, Y), axis=0)
    S = MatConvert(S, device, dtype)
    Fea = model_u(S)
    n = X.shape[0]
    return MMD_General(Fea, n, S, sigma, sigma0_u, ep, L=L)

def save_model(n,model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst,epoch=0):
    path = './checkpoint%d/'%n+str(epoch)+'/'
    try:
        os.makedirs(path) 
    except:
        pass
    torch.save(model.state_dict(), path+'model.pt')
    torch.save(epsilonOPT, path+'epsilonOPT.pt')
    torch.save(sigmaOPT, path+'sigmaOPT.pt')
    torch.save(sigma0OPT, path+'sigma0OPT.pt')
    torch.save(eps, path+'eps.pt')
    torch.save(cst, path+'cst.pt')

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

# run_saved_model(3000,500)
def run_saved_model(n, m, epoch=0, N=100):
    print('n =',n)

    model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst = load_model(n,epoch)
    model.eval()
    ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
    sigma = sigmaOPT ** 2
    sigma0_u = sigma0OPT ** 2
    print('cst',cst)

    ######### p value #########
    print('-------------------sigma--------------------')

    # compute mean and variance of sum(phi(Zi)) when Z~P
    M = N # evaluate the mean and variance of T~H0
    samples = np.zeros(M)
    with torch.torch.no_grad():                           
        for ii in trange(M):
            Z0 = dataset_P[np.random.choice(dataset_P.shape[0], 1100)]
            X1, Y1 = gen_fun(min(n,10000))
            mmd_PZ = mmdG(X1, Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
            mmd_QZ = mmdG(Y1, Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst

            # if n>10000:
            #     mmd_PZ = mmdG(X1[np.random.choice(n,10000)], Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
            #     mmd_QZ = mmdG(Y1[np.random.choice(n,10000)], Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
            # else:
            #     mmd_PZ = mmdG(X1, Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
            #     mmd_QZ = mmdG(Y1, Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
            samples[ii] = mmd_QZ-mmd_PZ 
    mean = np.mean(samples)
    var = np.var(samples)*M/(M-1)

    # Z~0.9P+0.1Q
    pval_list = np.zeros(N)
    for k in range(N):
        background_events = dataset_P[np.random.choice(dataset_P.shape[0], 1000)]
        signal_events = dataset_Q[np.random.choice(dataset_Q.shape[0], 100)]
        Z = np.concatenate((signal_events, background_events), axis=0)
        with torch.torch.no_grad():     
            X1, Y1 = gen_fun(min(n,10000))
            mmd_PZ = mmdG(X1, Z, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
            mmd_QZ = mmdG(Y1, Z, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
            T = mmd_QZ-mmd_PZ # test statistic
        pval_list[k] = -(T-mean)/np.sqrt(var) # p-value is in #sigma

    p_value = np.mean(pval_list)
    p_value_var = np.var(pval_list)
    print('----------------------------------')
    print('p-value = ({}-{})/{}'.format(T.item(),mean,np.sqrt(var)))
    print('p_value =', p_value)
    print('p_value_var =', p_value_var)
    print('----------------------------------')
    return p_value

def plot_train(n):
    J_star_u = np.load('./checkpoint%d/J_star_u.npy'%n)
    mmd_val_record = np.load('./checkpoint%d/mmd_val_record.npy'%n)
    mmd_var_record = np.load('./checkpoint%d/mmd_var_record.npy'%n)
    mmd_val_validations = np.load('./checkpoint%d/mmd_validations.npy'%n)
    J_validations = np.load('./checkpoint%d/J_validations.npy'%n)
    kk=0
    t=300
    plt.plot(range(t), J_star_u[kk, :][0:t], label='J')
    plt.plot(range(t), mmd_val_record[kk, 0:t], label='MMD')
    plt.plot(np.arange(301)[mmd_val_validations!=0], mmd_val_validations[mmd_val_validations!=0], label='MMD_validation')
    plt.plot(np.arange(301)[J_validations!=0] , J_validations[J_validations!=0], label='J_valid')
    plt.legend()
    plt.savefig('./checkpoint%d/loss-epoch.png'%n)
    plt.show()
    plt.clf()

def plot_m(n,epoch,M=100,m_list=np.arange(70,140,10)):
    model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst = load_model(n,epoch)
    model.eval()
    ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
    sigma = sigmaOPT ** 2
    sigma0_u = sigma0OPT ** 2
    print('ep',ep)
    print('sigma',sigma)
    print('sigma0_u',sigma0_u)
    print('cst',cst)

    ######### p value #########
    print('-------------------sigma--------------------')

    # compute mean and variance of sum(phi(Zi)) when Z~P
    # evaluate the mean and variance of T~H0
    MM = M*4
    samples = np.zeros(MM)
    with torch.torch.no_grad():                           
        for ii in trange(MM):
            Z0 = dataset_P[np.random.choice(dataset_P.shape[0], 1100)]
            mmd_PZ = mmdG(dataset_P[np.random.choice(n,10000)], Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
            mmd_QZ = mmdG(dataset_Q[np.random.choice(n,10000)], Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
            samples[ii] = mmd_QZ-mmd_PZ 
    mean = np.mean(samples)
    var = np.var(samples)*MM/(MM-1)
    print('mean =', mean)
    print('var =', var)
    print('sigma of mean =', np.sqrt(var/M))

    # Z~0.9P+0.1Q
    all_pval = np.zeros(len(m_list))
    for i,m in enumerate(m_list):
        print(m)
        pval_list = np.zeros(M)
        for k in trange(M):
            background_events = dataset_P[np.random.choice(dataset_P.shape[0], 1000)]
            signal_events = dataset_Q[np.random.choice(dataset_Q.shape[0], m)]
            Z = np.concatenate((signal_events, background_events), axis=0)
            with torch.torch.no_grad():     
                mmd_PZ = mmdG(dataset_P[np.random.choice(n,10000)], Z, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
                mmd_QZ = mmdG(dataset_Q[np.random.choice(n,10000)], Z, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
                T = mmd_QZ-mmd_PZ # test statistic
            pval_list[k] = -(T-mean)/np.sqrt(var) # p-value is in #sigma
        p_value = np.mean(pval_list)
        p_value_var = np.var(pval_list)
        all_pval[i] = p_value
        print('p_value =', p_value)
    print('m_list =', m_list)
    print('all_pval =', all_pval)
    plt.plot(m_list,all_pval)
    plt.savefig('./checkpoint%d/pval-m.png'%n)
    print('saved at'+ './checkpoint%d/pval-m.png'%n)
    return plt.show()


def plot_m_use_pi(n, epoch, 
                  m = 1000,
                  true_pi_list = np.arange(0, 0.2, 0.01),
                  test_pi_list = np.arange(0, 0.2, 0.01),
                  M = 100):
    model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst = load_model(n,epoch)
    model.eval()
    ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
    sigma = sigmaOPT ** 2
    sigma0_u = sigma0OPT ** 2
    print('n',n)

    ######### p value #########
    print('-------------------sigma--------------------')
    for i,m in enumerate(m_list):
        # compute mean and variance of sum(phi(Zi)) when Z~P
        # evaluate the mean and variance of T~H0
        samples = np.zeros(M)
        with torch.torch.no_grad():                           
            for ii in trange(M):
                Z0 = dataset_P[np.random.choice(dataset_P.shape[0], m)]
                #X1, Y1 = gen_fun(min(n,10000))
                if n>10000:
                    mmd_PZ = mmdG(X1[np.random.choice(n,10000)], Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
                    mmd_QZ = mmdG(Y1[np.random.choice(n,10000)], Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
                else:
                    mmd_PZ = mmdG(X1, Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
                    mmd_QZ = mmdG(Y1, Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
                samples[ii] = mmd_QZ-mmd_PZ 
        mean = np.mean(samples)
        var = np.var(samples)*M/(M-1)
        MMD2XY = mean

        # 计算 MMD²(Y,Z)-MMD²(X,Z) = 2 ⟨Z, X-Y⟩ + ⟨Y, Y⟩ - ⟨X, X⟩ 
        pi_true = true_pi_list[i]
        pi_test = pi_true/2
        T_list = np.zeros(M)
        for k in trange(M):
            background_events = dataset_P[np.random.choice(dataset_P.shape[0], m)]
            signal_events = dataset_Q[np.random.choice(dataset_Q.shape[0], m*pi_true)]
            Z = np.concatenate((signal_events, background_events), axis=0)
            with torch.torch.no_grad():     
                mmd_PZ = mmdG(X1[np.random.choice(n,10000)], Z, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
                mmd_QZ = mmdG(Y1[np.random.choice(n,10000)], Z, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
            T = mmd_QZ-mmd_PZ # test statistic
        p_value = np.mean(pval_list)
        p_value_var = np.var(pval_list)
        all_pval[i] = p_value
        print('p_value =', p_value)
    print(all_pval)
    plt.plot(m_list,all_pval)
    plt.savefig('./checkpoint%d/pval-m.png'%n)
    print('saved at ./checkpoint%d/pval-m.png'%n)
    return plt.show()

if __name__ == "__main__":

    n = int(sys.argv[1])
    epoch = int(sys.argv[2])

    # load data, please use .npy ones (40000), .gz (10999999)(11e6) is too large.
    dataset = np.load('HIGGS.npy')
    print('signal : background =',np.sum(dataset[:,0]),':',dataset.shape[0]-np.sum(dataset[:,0]))
    print('signal :',np.sum(dataset[:,0])/dataset.shape[0]*100,'%')
    # split into signal and background
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)
    # define a function to generate data
    def gen_fun(n):
        X = dataset_P[np.random.choice(dataset_P.shape[0], n)]
        Y = dataset_Q[np.random.choice(dataset_Q.shape[0], n)]
        return X, Y
    # set seed
    random_seed = 42
    

    import gc
    seed=42
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    dtype = torch.float
    device = torch.device("cuda:0")

    X1= dataset_P[n:2*n]
    Y1= dataset_Q[n:2*n]

    run_saved_model(n, 100, epoch, N=100)
    print('#####start plot#####')
    plot_m(n,epoch)

    # p_s = np.zeros(301)
    # for i in range(301):
    #     try:
    #         p_s[i] = run_saved_model(n, 100, i, N=20)
    #         print(i, p_s[i])
    #         plt.scatter(range(301),p_s)
    #     except:
    #         pass
    #     if p_s[i] != 0:
    #         plt.savefig('./checkpoint'+str(n)+'/p_s.png')

    