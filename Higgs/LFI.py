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

"""
We selected a five-layer neural network with 300 hidden units in each layer, 
a learning rate of 0.05, and a weight decay coefficient of 1e-5.
"""
class DN(torch.nn.Module):
    def __init__(self):
        super(DN, self).__init__()
        self.restored = False
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28, 300, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 20, bias=True),
        )
    def forward(self, input):
        output = self.model(input)
        return output
class DN(torch.nn.Module):
    def __init__(self):
        super(DN, self).__init__()
        self.restored = False
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28, 20, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 20, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 20, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 20, bias=True),
        )
    def forward(self, input):
        output = self.model(input)
        return output

def crit(mmd_val, mmd_var, liuetal=True, Sharpe=False):
    """compute the criterion, liu or Sharpe"""
    ######IMPORTANT: if we want to maximize, need to multiply by -1######
    return mmd_val - mmd_var
    # if liuetal:
    #     mmd_std_temp = torch.sqrt(mmd_var+10**(-8)) #this is std
    #     return torch.div(mmd_val, mmd_std_temp)
    # elif Sharpe:
    #     return mmd_val - 2.0 * mmd_var

# calculate the MMD for m!=n
def mmdG(X, Y, model_u, n, sigma, sigma0_u, device, dtype, ep):
    S = np.concatenate((X, Y), axis=0)
    S = MatConvert(S, device, dtype)
    Fea = model_u(S)
    n = X.shape[0]
    return MMD_General(Fea, n, S, sigma, sigma0_u, ep)

def save_model(model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst,epoch=0):
    path = './checkpoint/'+str(epoch)+'/'
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

def load_model(model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst,epoch=0):
    path = './checkpoint/'+str(epoch)+'/'
    model.load_state_dict(torch.load(path+'model.pt'))
    epsilonOPT = torch.load(path+'epsilonOPT.pt')
    sigmaOPT = torch.load(path+'sigmaOPT.pt')
    sigma0OPT = torch.load(path+'sigma0OPT.pt')
    eps = torch.load(path+'eps.pt')
    cst = torch.load(path+'cst.pt')
    return model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst

def train(n, m_list, title='Default', learning_rate=5e-4, 
            K=10, N=1000, N_epoch=50, print_every=100, batch_size=32, 
            test_on_new_sample=True, SGD=True, gen_fun=None, seed=42):  
    #set random seed for torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)
    # set torch
    torch.backends.cudnn.deterministic = True
    dtype = torch.float
    device = torch.device("cuda:0")
    #cuda.select_device(0)
    # set SGD
    if not SGD:
        batch_size = n
        batches = 1
    else:
        batches = n//batch_size + 1 # last batch could be empty
        n = batches*batch_size  
    #region
    # save parameters
    # parameters={'n':n,
    #             'm_list':m_list,
    #             'N_epoch':N_epoch,
    #             'learning_rate':learning_rate,
    #             'batch_size':batch_size,
    #             'batches':batches,
    #             'test_on_new_sample':test_on_new_sample,
    #             'SGD':SGD,
    #             'gen_fun':gen_fun(-1),
    #             'K':K,
    #             'seed' : seed,
    #             'N':N,}
    # with open('./data/PARAMETERS_'+title, 'wb') as pickle_file:
    #     pickle.dump(parameters, pickle_file)
    # print starting flag
    #endregion 
    print("\n------------------------------------------------")
    print("----- Starting K=%d independent kernels   -----"%(N_epoch))
    print("----- N_epoch=%d epochs per data trial    ------"%(K))
    print("----- N=%d tests per inference of Z per m -----"%(N))
    print("------------------------------------------------\n")
    #region
    # if test_on_new_sample:
    #     print("We test on new samples x, y not during t-raining")
    # else:
    #     print("We reuse samples x, y during training")
    # create arrays to store results
    #endregion
    Results = np.zeros([2, K, len(m_list)]) ###Result[{0, 1}, K, m] where K is an index
    J_star_u = np.zeros([K, N_epoch])
    mmd_val_record = np.zeros([K, N_epoch])
    mmd_var_record = np.zeros([K, N_epoch])

    ep_OPT = np.zeros([K])
    s_OPT = np.zeros([K])
    s0_OPT = np.zeros([K])

    for kk in range(K):
    # start a new kernel
        # prepare training data
        X, Y = gen_fun(n)
        total_S = [(X[i*batch_size:(i+1)*batch_size], 
                  Y[i*batch_size:(i+1)*batch_size]) 
                  for i in range(batches)]
        total_S = [MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in total_S]
        # prepare NN
        model = DN().cuda()
        # prepare other parameters
        epsilonOPT = MatConvert(np.random.rand(1) * (10 ** (-10)), device, dtype)
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.sqrt(np.random.rand(1) * 0.3), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.sqrt(np.random.rand(1) * 0.002), device, dtype)
        sigma0OPT.requires_grad = True
        eps=MatConvert(np.zeros((1,)), device, dtype)
        eps.requires_grad = True
        cst=MatConvert(np.ones((1,)), device, dtype) # set to 1 to meet liu etal objective
        cst.requires_grad = True
        # prepare optimizer
        optimizer_u = torch.optim.Adam(list(model.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT]+[eps]+[cst], lr=learning_rate)
        begin_time = time.time()
        for t in range(N_epoch):
            for ind in range(batches):
                # calculate parameters
                ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2
                # load training data from S
                S = total_S[ind]
    #region

                # Generate Higgs (P,Q)
                # N1_T = dataX.shape[0]
                # N2_T = dataY.shape[0]
                # np.random.seed(seed=1102 * kk + n)
                # ind1 = np.random.choice(N1_T, n, replace=False)
                # np.random.seed(seed=819 * kk + n)
                # ind2 = np.random.choice(N2_T, n, replace=False)
                # s1 = dataX[ind1,:4]
                # s2 = dataY[ind2,:4]
                # N1 = n
                # N2 = n
                # S = np.concatenate((s1, s2), axis=0)
                # S = MatConvert(S, device, dtype)
    #endregion
                # input into NN
                modelu_output = model(S) 
                # calculate MMD
                TEMP = MMDu(modelu_output, batch_size, S, sigma, sigma0_u, ep, cst) # could raise error
                # calculate objective
                mmd_val = -1 * TEMP[0]
                mmd_var = TEMP[1]
                STAT_u = crit(mmd_val, mmd_var) 
    #region
                # mmd_value_temp = -1 * (TEMP[0]+10**(-8))  # 10**(-8)
                # mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))  # 0.1
                # if mmd_std_temp.item() == 0:
                #     print('error!!')
                # if np.isnan(mmd_std_temp.item()):
                #     print('error!!')
                # STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
    #endregion
                J_star_u[kk, t] = STAT_u.item()
                mmd_val_record[kk, t] = mmd_val.item()
                mmd_var_record[kk, t] = mmd_var.item()
                # update parameters
                optimizer_u.zero_grad()
                STAT_u.backward(retain_graph=True)
                optimizer_u.step()
            
            # Print MMD, std of MMD and J
            if t % print_every == 0:
                time_per_epoch = (time.time() - begin_time)/print_every
                print('------------------------------------')
                print('Epoch:', t)
                print("mmd_value: ", mmd_val.item())
                print("mmd_var: ", mmd_var.item())
                print("Objective: ", STAT_u.item())
                print("time_per_epoch: ", time_per_epoch)
                #print(cst)
                print('------------------------------------')
                begin_time = time.time()
            if t%1000==0:
                save_model(model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst,t)
        
        plt.plot(range(N_epoch), J_star_u[kk, :], label='J')
        plt.plot(range(N_epoch), mmd_val_record[kk, :], label='MMD')
        plt.savefig('./checkpoint/loss-epoch.png')
        np.save('./checkpoint/J_star_u.npy', J_star_u)
        np.save('./checkpoint/mmd_val_record.npy', mmd_val_record)
        np.save('./checkpoint/mmd_var_record.npy', mmd_var_record)
        ep_OPT[kk] = ep.item()
        s_OPT[kk] = sigma.item()
        s0_OPT[kk] = sigma0_u.item()
        
        ''''''''''''''''''
        ''''''''''''''''''
        # testing how model behaves on untrained data
        # print test MMD, MMD_var and J
        print('CRITERION ON NEW SET OF DATA:')            
        X1, Y1 = gen_fun(n)
        
        with torch.torch.no_grad():
            S1 = np.concatenate((X1, Y1), axis=0)
            S1 = MatConvert(S1, device, dtype)
            modelu_output = model(S1)
            TEMP = MMDu(modelu_output, n, S1, sigma, sigma0_u, ep, cst)
            mmd_value_temp, mmd_var_temp = TEMP[0], TEMP[1]
            STAT_u = crit(mmd_value_temp, mmd_var_temp)
            if True:
                print("TEST mmd_value: ", mmd_value_temp.item()) 
                print("TEST Objective: ", STAT_u.item())
    #region

        # generate Z, calculate P(success|Z~X,Y)
        # H_x = np.zeros(N) 
        # H_y = np.zeros(N)
        # print("Under this trained kernel, we run N = %d times LFI: "%N)
        # if test_on_new_sample:
        #     X, Y = gen_fun(n)
        # # run through all m do LFI
        # for i in range(len(m_list)):
        #     m = m_list[i]
        #     print("start testing m = %d"%m)
        #     for k in range(N):     
        #         #t=time.time()  
        #         Z1, Z2 = gen_fun(m)
        #         #print(cst)
        #         #print(time.time()-t)
        #         mmd_XZ = mmdG(X, Z1, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
        #         mmd_YZ = mmdG(Y, Z1, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
        #         H_x[k] = mmd_XZ < mmd_YZ    
        #         mmd_XZ = mmdG(X, Z2, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst 
        #         mmd_YZ = mmdG(Y, Z2, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
        #         H_y[k] = mmd_XZ > mmd_YZ
        #     Results[0, kk, i] = H_x.sum() / float(N)
        #     Results[1, kk, i] = H_y.sum() / float(N)
        #     print('------------------------------------------')
        #     print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~X): ", Results[0, kk, i])
        #     print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~Y): ", Results[1, kk, i])
        #     print('------------------------------------------')
    #endregion
        
        ''''''''''''''''''
        ''''''''''''''''''
        # compute #sigma
        print('-------------------sigma--------------------')
        gpu_usage()                             
        X, Y = gen_fun(n)
        signal_events = dataset_P[np.random.choice(dataset_P.shape[0], 100)]
        background_events = dataset_Q[np.random.choice(dataset_Q.shape[0], 1000)]
        Z = np.concatenate((signal_events, background_events), axis=0)
        mmd_PZ = mmdG(X, Z, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
        mmd_QZ = mmdG(Y, Z, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
        T = mmd_QZ-mmd_PZ # test statistic
        # compute mean and variance
        samples = np.zeros(1000)
        #torch.cuda.empty_cache()
        # cuda.select_device(0)
        # cuda.close()
        # cuda.select_device(0)
        #gpu_usage()  
        with torch.torch.no_grad():                           
            for ii in range(1000):
                Z0 = dataset_P[np.random.choice(dataset_P.shape[0], 1100)]
                mmd_PZ = mmdG(X, Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
                mmd_QZ = mmdG(Y, Z0, model, n, sigma, sigma0_u, device, dtype, ep)[0] * cst
                samples[ii] = mmd_QZ-mmd_PZ 
        mean = np.mean(samples)
        var = np.var(samples)*1000/999
        p_value = (T-mean)/np.sqrt(var) # p-value is in #sigma
        print('p-value =', p_value)
    #LFI_plot(n_list, title=title)

if __name__ == "__main__":
    # load data, please use .npy ones (40000), .gz (10999999)(11e6) is too large.
    if False:
        df = pd.read_csv('HIGGS.csv.gz') # FROM http://archive.ics.uci.edu/ml/datasets/HIGGS
        print('df.shape =', df.shape)
        dataset = df.to_numpy()
    else:
        dataset = np.load('HIGGS_400000.npy')
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

    # load title
    try:
        title = sys.argv[1]
    except:
        print("Warning: No title given, using default")
        title = 'untitled_run'
    # region
    # data = pickle.load(open('./HIGGS_TST.pckl', 'rb'))
    # dataX = data[0]
    # dataY = data[1]
    # print(dataX.shape, dataY.shape)
    #endregion
    # run 
    n = 10000
    m_list = [400] # 不做LFI没有用
    train(n, [50], 
        title = title, 
        K = 1, 
        N = 100, # 不做LFI没有用
        N_epoch = 10, 
        print_every = 10, 
        batch_size = 128, 
        learning_rate =5e-4, 
        test_on_new_sample = True, 
        SGD = True, 
        gen_fun = gen_fun, 
        seed = random_seed)