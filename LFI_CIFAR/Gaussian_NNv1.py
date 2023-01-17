import numpy as np
import torch
from sklearn.utils import check_random_state
from utils import *
from matplotlib import pyplot as plt
import torch.nn as nn
dtype = torch.float
device = torch.device("cuda:0")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic=True
dtype = torch.float
device = torch.device("cuda:0")
import time
# limit to 1 core
torch.set_num_threads(1)


class ConvNet_CIFAR10(nn.Module):
    def __init__(self):
        super(ConvNet_CIFAR10, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block =([nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=True),  
                     nn.Dropout2d(0)])
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        self.model = nn.Sequential(
            nn.Unflatten(1,(3,32,32)),
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        ds_size = 2
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 300))
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)
        return feature
    
def crit(mmd_val, mmd_var, liuetal=True, Sharpe=False):
    if liuetal:
        mmd_std_temp = torch.sqrt(mmd_var+10**(-8)) #this is std
        return -1 * torch.div(mmd_val, mmd_std_temp)
    elif Sharpe:
        return mmd_val - 2.0 * mmd_var

def mmdGS(X, Y, model_u, n, sigma, cst, device, dtype):
    #a = time.time()
    S = np.concatenate((X, Y), axis=0)
    S = MatConvert(S, device, dtype)
    #b = time.time()
    Fea = model_u(S)
    n = X.shape[0]
    mmd =  MMDs(Fea, n, S, sigma, cst, is_var_computed=False)
    #c = time.time()
    # print('MatConvert time', b-a)
    # print('MMD time', c-b)
    # print('_________')
    return mmd

def load_diffusion_cifar_32():
    diffusion = np.load("../Diffusion/ddpm_generated_images2.npy").transpose(0,3,1,2)
    cifar10 = np.load('../data/cifar_data.npy')
    dataset_P = diffusion.reshape(diffusion.shape[0], -1)
    dataset_Q = cifar10.reshape(cifar10.shape[0], -1)
    return dataset_Q, dataset_P

DP, DQ = load_diffusion_cifar_32()
DP1 = DP[:4000, :]
DQ1 = DQ[:4000, :]
DP2 = DP[4000:, :]
DQ2 = DQ[4000:, :]
print(DP1.shape, DQ1.shape)
def gen_fun1(n):
    X = DP1[np.random.choice(DP1.shape[0], n, replace=False), :]
    Y = DQ1[np.random.choice(DQ1.shape[0], n, replace=False), :]
    return X, Y
def gen_fun2(n):
    X = DP2[np.random.choice(DP2.shape[0], n, replace=False), :]
    Y = DQ2[np.random.choice(DQ2.shape[0], n, replace=False), :]
    return X, Y

def train_d(n, learning_rate=5e-4, N=1000, N_epoch=50, print_every=20, batch_size=32):  
    batches=n//batch_size
    assert n%batch_size==0
    print("##### Starting N_epoch=%d epochs per data trial #####"%(N_epoch))
    if True:
        X, Y = gen_fun1(n)
        total_S=[(X[i*batch_size:i*batch_size+batch_size], Y[i*batch_size:i*batch_size+batch_size]) for i in range(batches)]
        total_S=[MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in total_S]
        model_u = ConvNet_CIFAR10().cuda()
        sigmaOPT = MatConvert(np.sqrt(np.array([0.1])), device, dtype)
        sigmaOPT.requires_grad = True
        cst=MatConvert(np.ones((1,)), device, dtype) # set to 1 to meet liu etal objective
        cst.requires_grad = False
        optimizer_u = torch.optim.Adam(list(model_u.parameters())+[sigmaOPT], lr=learning_rate)
        for t in range(N_epoch):
            for ind in range(batches):
                sigma = sigmaOPT ** 2
                S=total_S[ind]
                modelu_output = model_u(S) 
                TEMP = MMDs(modelu_output, batch_size, S, sigma, cst)
                STAT_u = crit(TEMP[0], TEMP[1])
                optimizer_u.zero_grad()
                STAT_u.backward(retain_graph=True)
                optimizer_u.step()
            if t % print_every == 0:
                print('------------------------------------')
                print('Epoch:', t)
                print("mmd_value: ", TEMP[0].item()) 
                print("Objective: ", STAT_u.item())
                print('------------------------------------')
        print('CRITERION ON NEW SET OF DATA:')            
        X1, Y1 = gen_fun2(n)
        with torch.torch.no_grad():
            S1 = np.concatenate((X1, Y1), axis=0)
            S1 = MatConvert(S1, device, dtype)
            modelu_output = model_u(S1)
            TEMP = MMDs(modelu_output, n, S1, sigma, cst)
            mmd_value_temp, mmd_var_temp = TEMP[0], TEMP[1]
            STAT_u = crit(mmd_value_temp, mmd_var_temp)
            if True:
                print("TEST mmd_value: ", mmd_value_temp.item()) 
                print("TEST Objective: ", STAT_u.item())
        return model_u, sigmaOPT**2
k=1
for k in range(1, 11):
    n=512
    N=500
    model_u, sigma=train_d(n, learning_rate=5e-4, N=500, N_epoch=100, batch_size=32)
    m_list=list(range(2, 62, 2))
    cst = 1.0
    F1 = './data/1result_MMDG_'+str(n)+'_'+str(k)+'.txt'
    print('--------------------------')
    print(' Start Testing n=%d'%n)
    print('--------------------------')
    fwrite('', F1, message='New File '+str(n))
    if True:
        N_f = float(N)
        with torch.no_grad():
            H_u = np.zeros(N) 
            H_v = np.zeros(N)
            R_u = np.zeros(N)
            R_v = np.zeros(N)
            print("Under this trained kernel, we run N = %d times LFI: "%N)
            for i in range(len(m_list)):
                print("start testing m = %d"%m_list[i])
                m = m_list[i]    
                for k in range(N):     
                    X , Y  = gen_fun1(n)
                    XX1, YY1 = gen_fun2(n+m)
                    X1     = XX1[:n]
                    Z1, Z2 = XX1[n:], YY1[n:] 
                    stat=[]
                    for j in range(100):
                        Z_temp = X1[np.random.permutation(n)][:m]
                        mmd_XZ = mmdGS(X, Z_temp, model_u, n, sigma, cst, device, dtype)[0] 
                        mmd_YZ = mmdGS(Y, Z_temp, model_u, n, sigma, cst, device, dtype)[0]
                        stat.append(float(mmd_XZ - mmd_YZ))
                    thres = np.percentile(stat, 95)
                    #print(k)
                    if k%50==0:
                        print('threshold 95 is:', thres, k)
                    mmd_XZ = mmdGS(X, Z1, model_u, n, sigma, cst, device, dtype)[0]
                    mmd_YZ = mmdGS(Y, Z1, model_u, n, sigma, cst, device, dtype)[0]
                    H_u[k] = mmd_XZ - mmd_YZ < 0.0
                    R_u[k] = mmd_XZ - mmd_YZ < thres+1e-10
                    mmd_XZ = mmdGS(X, Z2, model_u, n, sigma, cst, device, dtype)[0]
                    mmd_YZ = mmdGS(Y, Z2, model_u, n, sigma, cst, device, dtype)[0]
                    H_v[k] = mmd_XZ - mmd_YZ > 0.0
                    R_v[k] = mmd_XZ - mmd_YZ > thres+1e-10
                st1="n, m= "+str(n)+str('  ')+str(m)+" --- P(max|Z~X):  "+str(H_u.sum()/N_f)
                st2="n, m= "+str(n)+str('  ')+str(m)+" --- P(max|Z~Y):  "+str(H_v.sum()/N_f)
                st3="n, m= "+str(n)+str('  ')+str(m)+" --- P(95|Z~X):  "+str(R_u.sum()/N_f)
                st4="n, m= "+str(n)+str('  ')+str(m)+" --- P(95|Z~Y):  "+str(R_v.sum()/N_f)
                fwrite(st1, F1)
                fwrite(st2, F1)
                fwrite(st3, F1)
                fwrite(st4, F1)
                print('k, i, m = ', k, i, m)
                print('------------------------------------')



import torch
A = torch.tensor([[1, 2, 3], [4, 5, 6]])