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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used
"""
We selected a five-layer neural network with 300 hidden units in each layer, 
a learning rate of 0.05, and a weight decay coefficient of 1e-5.
"""
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
            #torch.nn.Dropout(p=0.5),
        )
    def forward(self, input):
        output = self.model(input)
        return output

class another_DN(torch.nn.Module):
    def __init__(self):
        super(another_DN, self).__init__()
        self.restored = False
        self.model = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, x_in, bias=True),
        )
    def forward(self, input):
        output = self.model(input) + input
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
def mmdG(X, Y, model_u, n, sigma, sigma0_u, device, dtype, ep):
    S = np.concatenate((X, Y), axis=0)
    S = MatConvert(S, device, dtype)
    Fea = model_u(S)
    n = X.shape[0]
    return MMD_General(Fea, n, S, sigma, sigma0_u, ep)

def save_model(n,model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst,epoch=0):
    path = './checkpoint%d/'%n+str(epoch)+'/'
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
def run_saved_model(n, m, epoch=0, N=10):
    n_backup = n
    n = min(n,10000)
    print('n =',n)
    device = torch.device("cuda:0")
    dtype = torch.float32
    model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst = load_model(n_backup,epoch)
    model.eval()
    ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
    sigma = sigmaOPT ** 2
    sigma0_u = sigma0OPT ** 2
    print('cst',cst)
    X1= dataset_P[n_backup:n_backup+n]
    Y1= dataset_Q[n_backup:n_backup+n]
    print('Test LFI')
    m_list=[m]
    K=1
    kk=0
    # generate Z, calculate P(success|Z~X,Y)
    H_x = np.zeros(N) 
    H_y = np.zeros(N)
    print("Under this trained kernel, we run N = %d times LFI: "%N)
    # run through all m do LFI
    Results = np.zeros([2, K, len(m_list)]) ###Result[{0, 1}, K, m] where K is an index
    for i in range(len(m_list)):
        m = m_list[i]
        print("start testing m = %d"%m)
        for k in trange(N):     
            #t=time.time()  
            Z1, Z2 = gen_fun(m)
            #print(cst)
            #print(time.time()-t)
            with torch.torch.no_grad():
                mmd_XZ = mmdG(X1, Z1, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
                mmd_YZ = mmdG(Y1, Z1, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
                H_x[k] = mmd_XZ < mmd_YZ    
                mmd_XZ = mmdG(X1, Z2, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst 
                mmd_YZ = mmdG(Y1, Z2, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
                H_y[k] = mmd_XZ > mmd_YZ
        Results[0, kk, i] = H_x.sum() / float(N)
        Results[1, kk, i] = H_y.sum() / float(N)
        print('------------------------------------------')
        print("n, m=",str(n_backup)+str('  ')+str(m),"--- P(success|Z~X): ", Results[0, kk, i])
        print("n, m=",str(n_backup)+str('  ')+str(m),"--- P(success|Z~Y): ", Results[1, kk, i])
        print('------------------------------------------')
    ######### p value #########
    print('-------------------sigma--------------------')
    gpu_usage()                             

    # compute mean and variance of sum(phi(Zi)) when Z~P
    M = 100 # evaluate the mean and variance of T~H0
    samples = np.zeros(M)
    with torch.torch.no_grad():                           
        for ii in trange(M):
            Z0 = dataset_P[np.random.choice(dataset_P.shape[0], 1100)]
            X1, Y1 = gen_fun(min(n,20000))
            mmd_PZ = mmdG(X1, Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
            mmd_QZ = mmdG(Y1, Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
            samples[ii] = mmd_QZ-mmd_PZ 
    mean = np.mean(samples)
    var = np.var(samples)

    # Z~0.9P+0.1Q
    pval_list = np.zeros(N)
    for k in range(N):
        background_events = dataset_P[np.random.choice(dataset_P.shape[0], 1000)]
        signal_events = dataset_Q[np.random.choice(dataset_Q.shape[0], 100)]
        Z = np.concatenate((signal_events, background_events), axis=0)
        with torch.torch.no_grad():     
            X1, Y1 = gen_fun(min(n,20000))
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

def train(n, m_list, title='Default', learning_rate=5e-4, 
            K=10, N=1000, N_epoch=50, print_every=100, batch_size=32, 
            SGD=True, gen_fun=None, seed=42,
            dataset_P=None, dataset_Q=None,
            load_epoch=0, load_n=0,
            step_size=10, gamma=0.1,
            momentum = 0.9, weight_decay=0.0,):  
    n_backup = n
    try:
        os.mkdir('./checkpoint%d'%n_backup)
    except:
        pass
    #set random seed for torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)
    # set torch
    torch.backends.cudnn.deterministic = True
    dtype = torch.float
    device = torch.device("cuda:0")
    #cuda.select_device(0)
    batches = (n-1)//batch_size + 1 # last batch could be empty
    n = batches*batch_size  
    print("\n------------------------------------------------")
    print("----- Starting K=%d independent kernels   -----"%(N_epoch))
    print("----- N_epoch=%d epochs per data trial    ------"%(K))
    print("----- N=%d tests per inference of Z per m -----"%(N))
    print("------------------------------------------------\n")
    J_star_u = np.zeros([K, N_epoch])
    mmd_val_record = np.zeros([K, N_epoch])
    mmd_var_record = np.zeros([K, N_epoch])
    ep_OPT = np.zeros([K])
    s_OPT = np.zeros([K])
    s0_OPT = np.zeros([K])
    X = dataset_P[0:n]
    Y = dataset_Q[0:n]
    print(X.shape, Y.shape)
    for kk in range(K):
    # start a new kernel
        # prepare training data
        total_S = [(X[i*batch_size:(i+1)*batch_size], 
                  Y[i*batch_size:(i+1)*batch_size]) 
                  for i in range(batches)]
        total_S = [MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in total_S]
        # prepare NN
        model = DN().cuda()
        another_model = another_DN().cuda()
        # prepare other parameters
        epsilonOPT = MatConvert(np.random.rand(1) * (10 ** (-10)), device, dtype)
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.sqrt(np.random.rand(1) * 0.3), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.sqrt(np.random.rand(1) * 0.002), device, dtype)
        sigma0OPT.requires_grad = True
        eps=MatConvert(np.zeros((1,)), device, dtype)
        eps.requires_grad = True
        cst=MatConvert(1*np.ones((1,)), device, dtype) # set to 1 to meet liu etal objective
        cst.requires_grad = False
        # load
        if load_epoch>0:
            model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst = load_model(load_n, load_epoch)
        # prepare optimizer
        params = list(model.parameters())+list(another_model.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT]+[eps]+[cst]
        if SGD:
            optimizer_u = torch.optim.SGD(params, lr=learning_rate,
                                     momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer_u = torch.optim.Adam(params, lr=learning_rate)
        begin_time = time.time()
        # validation data
        S1_v = np.concatenate((dataset_P[n + np.random.choice(n, min(n,10000), replace=False)], 
                                dataset_Q[n + np.random.choice(n, min(n,10000), replace=False)]), axis=0)
        S1_v = MatConvert(S1_v, device, dtype)
        J_validations = np.zeros([N_epoch])
        mmd_val_validations = np.zeros([N_epoch])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_size, gamma=gamma)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_u, T_max=10, eta_min=0)
        #############################
        #############################
        #############################
        for t in range(N_epoch):
            print(t)
            order = np.random.permutation(batches)
            for ind in tqdm(order):
                optimizer_u.zero_grad()
                # calculate parameters
                ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2
                # load training data from S
                S = total_S[ind]
                # input into NN
                modelu_output = model(S) 
                another_output = another_model(S)
                # calculate MMD
                TEMP = MMDu(modelu_output, batch_size, another_output, sigma, sigma0_u, ep, cst, L=L) # could raise error
                # calculate objective
                mmd_val = -1 * TEMP[0]
                mmd_var = TEMP[1]
                STAT_u = crit(mmd_val, mmd_var) 
                # update parameters
                STAT_u.backward(retain_graph=False)
                optimizer_u.step()      
            scheduler.step()
            #validation
            print('validation')
            with torch.torch.no_grad():
                modelu_output = model(S1_v)
                another_output = another_model(S1_v)
                TEMP = MMDu(modelu_output, min(n,10000), another_output, sigma, sigma0_u, ep, cst, L=L)
                mmd_value_temp, mmd_var_temp = -TEMP[0], TEMP[1]
                mmd_val_validations[t] = mmd_value_temp.item()
                J_validations[t] = crit(mmd_value_temp, mmd_var_temp).item()
                J_star_u[kk, t] = STAT_u.item()
                mmd_val_record[kk, t] = mmd_val.item()
                mmd_var_record[kk, t] = mmd_var.item()
            # Print MMD, std of MMD and J
            if t % print_every == 0:
                time_per_epoch = (time.time() - begin_time)/print_every# print
                print('------------------------------------')
                print('n:', n)
                print('Epoch:', t ,'+',load_epoch)
                print("mmd_value: ", mmd_val.item())
                print("mmd_var: ", mmd_var.item())
                print("Objective: ", STAT_u.item())
                print("time_per_epoch: ", time_per_epoch)
                print('------------------------------------')
                begin_time = time.time()
                save_model(n_backup,model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst,t+load_epoch)
                plt.plot(range(t), J_star_u[kk, :][0:t], label='J')
                plt.plot(range(t), mmd_val_record[kk, 0:t], label='MMD')
                plt.plot(np.arange(N_epoch)[mmd_val_validations!=0], mmd_val_validations[mmd_val_validations!=0], label='MMD_validation')
                plt.plot(np.arange(N_epoch)[J_validations!=0] , J_validations[J_validations!=0], label='J_valid')
                plt.title('%f'%(np.min(J_validations)))        
                plt.xlabel(t+load_epoch)
                plt.legend()
                plt.savefig('./checkpoint%d/loss-epoch.png'%n_backup)
                plt.show()
                plt.clf()
                #print('-------------start test------------------')
                #if t>0:
                #    run_saved_model(n_backup, 100, epoch=t, N=100)
        #############################
        #############################
        #############################
        #############################
                np.save('./checkpoint%d/J_star_u.npy'%n_backup, J_star_u)
                np.save('./checkpoint%d/mmd_val_record.npy'%n_backup, mmd_val_record)
                np.save('./checkpoint%d/mmd_var_record.npy'%n_backup, mmd_var_record)
                np.save('./checkpoint%d/mmd_validations.npy'%n_backup, mmd_val_validations)
                np.save('./checkpoint%d/J_validations.npy'%n_backup, J_validations)
        ep_OPT[kk] = ep.item()
        s_OPT[kk] = sigma.item()
        s0_OPT[kk] = sigma0_u.item()

        print('################## Start test ##################')
        run_saved_model(n_backup, 100, (N_epoch//100)*100, N=10)

def standardize(X):
    for j in range(X.shape[1]):
        if j >0:
            vec = X[:, j]
            if np.min(vec) < 0:
                # Assume data is Gaussian or uniform -- center and standardize.
                vec = vec - np.mean(vec)
                vec = vec / np.std(vec)
            elif np.max(vec) > 1.0:
                # Assume data is exponential -- just set mean to 1.
                vec = vec / np.mean(vec)
            X[:,j] = vec
    return X

if __name__ == "__main__":
    # load data, please use .npy ones (40000), .gz (10999999)(11e6) is too large.
    dataset = np.load('HIGGS.npy')
    print('signal : background =',np.sum(dataset[:,0]),':',dataset.shape[0]-np.sum(dataset[:,0]))
    print('signal :',np.sum(dataset[:,0])/dataset.shape[0]*100,'%')
    # split into signal and background
    #dataset = standardize(dataset)
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)
    # randumly shuffle
    #np.random.shuffle(dataset_P)
    #np.random.shuffle(dataset_Q)

    # define a function to generate data
    def gen_fun(n):
        X = dataset_P[np.random.choice(dataset_P.shape[0], n)]
        Y = dataset_Q[np.random.choice(dataset_Q.shape[0], n)]
        return X, Y
    Mixture = dataset_P[np.random.choice(dataset_P.shape[0], dataset_P.shape[0], replace=False)]
    # print(len(np.arange(0,dataset_P.shape[0],10)))
    # print(dataset_P.shape[0]//10)
    nx = dataset_P.shape[0]
    prop = 10
    Mixture[np.arange(0,nx,prop)] = dataset_Q[np.random.choice(dataset_Q.shape[0], 1+(nx-1)//prop, replace=False)]
    # set seed
    random_seed = 42
    # load title
    try:
        title = sys.argv[1]
    except:
        print("Warning: No title given, using default")
        title = 'untitled_run'
   # n = 700001
    m_list = [800] # 不做LFI没有用
    N_epoch = 401

    for n in [3000, 6000, 10000, 30000, 60000, 2000000]:
        if n > 100000:
            N_epoch = 201; print_every = 5
        train(n, [50], 
            title = title, 
            K = 1, 
            N = 100, # 不做LFI没有用
            N_epoch = N_epoch, # 只load就设成1
            print_every = 10, 
            batch_size = min(n,1024), 
            learning_rate = 2e-3, 
            SGD = True, 
            gen_fun = gen_fun, 
            seed = random_seed,
            dataset_P = dataset_P, dataset_Q = dataset_Q, #Mixture
            load_epoch = 0, load_n=n,
            step_size=1, gamma=1,
            momentum=0.99, weight_decay=0.000)

    print('################## Start test ##################')
    gc.collect()
    torch.cuda.empty_cache()
    run_saved_model(n, 100, epoch=30, N=20)
    