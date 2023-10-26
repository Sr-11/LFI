import numpy as np
import torch
import sys
from sklearn.utils import check_random_state
from utils import *
from matplotlib import pyplot as plt
import pickle
import torch.nn as nn
import time
device = torch.device("cuda:0")
dtype = torch.float32

class ConvNet_CIFAR10(nn.Module):
    """
    input: (N,3x32x32)
    output: (N,300)
    """
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
        ds_size = 32 // 2 ** 4
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

def load_diffusion_cifar_32():
    diffusion = np.load("../Diffusion/ddpm_generated_images2.npy").transpose(0,3,1,2)
    cifar10 = np.load('../data/cifar_data.npy')
    dataset_P = diffusion.reshape(diffusion.shape[0], -1)
    dataset_Q = cifar10.reshape(cifar10.shape[0], -1)
    return dataset_P, dataset_Q[:10000, :], dataset_Q[10000:, :]

DP, DQ_1, DQ_2 = load_diffusion_cifar_32()
#print(DP.shape, DQ_1.shape, DQ_2.shape): (13140, 3072) (10000, 3072) (40000, 3072)
mix_rate=2 #For each DP, match with mix_rate*DQ data points

test_DP1=np.concatenate((DP[:2000, :], DQ_1[:4000, :]), axis=0)
test_DQ1=DQ_1[4000: 10000, :]

train_DP1=np.concatenate((DP[2000:7000, :], DQ_2[:10000, :]), axis=0)
train_DQ1=DQ_2[10000: 25000, :]

print(train_DP1.shape, train_DQ1.shape, test_DP1.shape, test_DQ1.shape)
#generate a random shuffle over train_DP1, print the first item
train_DP1 = train_DP1[np.random.choice(train_DP1.shape[0], train_DP1.shape[0], replace=False), :]
train_DQ1 = train_DQ1[np.random.choice(train_DQ1.shape[0], train_DQ1.shape[0], replace=False), :]
test_DP1 = test_DP1[np.random.choice(test_DP1.shape[0], test_DP1.shape[0], replace=False), :]
test_DQ1 = test_DQ1[np.random.choice(test_DQ1.shape[0], test_DQ1.shape[0], replace=False), :]

DP1_t = MatConvert(train_DP1, device, dtype)
DQ1_t = MatConvert(train_DQ1, device, dtype)
DP2_t = MatConvert(test_DP1, device, dtype)
DQ2_t = MatConvert(test_DQ1, device, dtype)

def gen_fun1(n): #n at most 15000
    X = train_DP1[np.random.choice(train_DP1.shape[0], n, replace=False), :]
    Y = train_DQ1[np.random.choice(train_DQ1.shape[0], n, replace=False), :]
    return X, Y
def gen_fun2(n): #n at most 6000
    X = test_DP1[np.random.choice(test_DP1.shape[0], n, replace=False), :]
    Y = test_DQ1[np.random.choice(test_DQ1.shape[0], n, replace=False), :]
    return X, Y

def gen_fun1_t(n):
    X = DP1_t[np.random.choice(DP1_t.shape[0], n, replace=False), :]
    Y = DQ1_t[np.random.choice(DQ1_t.shape[0], n, replace=False), :]
    return X, Y

def gen_fun2_t(n):
    X = DP2_t[np.random.choice(DP2_t.shape[0], n, replace=False), :]
    Y = DQ2_t[np.random.choice(DQ2_t.shape[0], n, replace=False), :]
    return X, Y

def mmdGST(X, Y, model_u, n, sigma, cst, device, dtype):
    #Same as mmdGS but X and Y are already in torch format
    S = torch.cat((X, Y), dim=0)
    Fea = model_u(S)
    n = X.shape[0]
    return MMDs(Fea, n, S, sigma, cst, is_var_computed=False)

def mmdGs(X_org, Y_org, sigma):
    Dxx_org = Pdist2(X_org, X_org)
    #print(Dxx_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    Kx = torch.exp(-Dxx_org / sigma)
    Ky = torch.exp(-Dyy_org / sigma)
    Kxy = torch.exp(-Dxy_org / sigma)
    return h1_mean_var_gram(Kx, Ky, Kxy, False)
def MMDus(len_s, Fea_org, sigma):
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    Kx = torch.exp(-Dxx_org / sigma)
    Ky = torch.exp(-Dyy_org / sigma)
    Kxy = torch.exp(-Dxy_org / sigma)
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=True)

def train_d(n, learning_rate=5e-4, N_epoch=1000, print_every=20):  
    print("##### Starting N_epoch=%d epochs per data trial #####"%(N_epoch))
    if True:
        X, Y = gen_fun1(n)
        S = np.concatenate((X, Y), axis=0)
        S = MatConvert(S, device, dtype)
        sigmaOPT = torch.tensor(5000.0).to(device, dtype)
        sigmaOPT.requires_grad = True
        optimizer_u = torch.optim.Adam([sigmaOPT], lr=learning_rate)
        for t in range(N_epoch):
                sigma = sigmaOPT ** 2
                TEMP = MMDus(n, S, sigma)
                #print("mmd_value: ", TEMP[0].item())
                STAT_u = crit(TEMP[0], TEMP[1]) 
                optimizer_u.zero_grad()
                STAT_u.backward(retain_graph=True)
                optimizer_u.step()
        return sigmaOPT**2, X, Y

N=300
N_f=float(N)
n=1920
m_list=[48, 96, 128, 160, 192, 216, 240, 256, 320, 384]
for _ in range(10):
            sigma, X, Y=train_d(n, learning_rate=5e-1, N_epoch=1000)
            H_u = np.zeros(N) 
            H_v = np.zeros(N)
            R_u = np.zeros(N)
            R_v = np.zeros(N)
            P_u = np.zeros(N)
            P_v = np.zeros(N)
            print("Under this trained kernel, we run N = %d times LFI: "%N)
            for i in range(len(m_list)):
                print("start testing m = %d"%m_list[i])
                m = m_list[i]    
                for k in range(N):
                    stat=[]
                    X_cal, _ = gen_fun2_t(n)
                    for j in range(100):
                        Z_temp = X_cal[np.random.choice(X_cal.shape[0], m, replace=False), :]
                        mmd_XZ = mmdGs(X, Z_temp, sigma)[0]
                        mmd_YZ = mmdGs(Y, Z_temp, sigma)[0]
                        stat.append(float(mmd_XZ - mmd_YZ))
                    stat = np.sort(stat)
                    thres = np.percentile(stat, 95)
                    Z1, Z2 = gen_fun2_t(m)
                    mmd_XZ = mmdGs(X, Z1, sigma)[0]
                    mmd_YZ = mmdGs(Y, Z1, sigma)[0]
                    H_u[k] = mmd_XZ - mmd_YZ < 0.0
                    R_u[k] = mmd_XZ - mmd_YZ < thres
                    P_u[k] = np.searchsorted(stat, float(mmd_XZ - mmd_YZ), side="right")/100.0
                    mmd_XZ = mmdGs(X, Z2, sigma)[0]
                    mmd_YZ = mmdGs(Y, Z2, sigma)[0]
                    H_v[k] = mmd_XZ - mmd_YZ > 0.0
                    R_v[k] = mmd_XZ - mmd_YZ > thres
                    P_v[k] = np.searchsorted(stat, float(mmd_XZ - mmd_YZ), side="right")/100.0
                print("n, m= "+str(n)+str('  ')+str(m)+" --- P(max|Z~X):  "+str(H_u.sum()/N_f))
                print("n, m= "+str(n)+str('  ')+str(m)+" --- P(max|Z~Y):  "+str(H_v.sum()/N_f))
                print("n, m= "+str(n)+str('  ')+str(m)+" --- P(95|Z~X):  "+str(R_u.sum()/N_f))
                print("n, m= "+str(n)+str('  ')+str(m)+" --- P(95|Z~Y):  "+str(R_v.sum()/N_f))
                print("n, m= "+str(n)+str('  ')+str(m)+" --- P(Ep|Z~X):  "+str(P_u.sum()/N_f))
                print("n, m= "+str(n)+str('  ')+str(m)+" --- P(Ep|Z~Y):  "+str(P_v.sum()/N_f))