import numpy as np
import torch
from sklearn.utils import check_random_state
from utils import *
from matplotlib import pyplot as plt
import torch.nn as nn
dtype = torch.float
device = torch.device("cuda:0")
seed = 4
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic=True
dtype = torch.float
device = torch.device("cuda:0")

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

def load_diffusion_cifar_32():
    diffusion = np.load("../Diffusion/ddpm_generated_images2.npy").transpose(0,3,1,2)
    cifar10 = np.load('../data/cifar_data.npy')
    dataset_P = diffusion.reshape(diffusion.shape[0], -1)
    dataset_Q = cifar10.reshape(cifar10.shape[0], -1)
    return dataset_P, dataset_Q

DP, DQ = load_diffusion_cifar_32()
DP1 = DP[:4000, :]
DQ1 = DQ[:4000, :]
DP2 = DP[4000:13140, :]
DQ2 = DQ[4000:13140, :]
print(DP1.shape, DQ1.shape)
print(DP2.shape, DQ2.shape)
DP1_t = MatConvert(DP1, device, dtype)
DQ1_t = MatConvert(DQ1, device, dtype)
DP2_t = MatConvert(DP2, device, dtype)
DQ2_t = MatConvert(DQ2, device, dtype)

def gen_fun1(n):
    X = DP1[np.random.choice(DP1.shape[0], n, replace=False), :]
    Y = DQ1[np.random.choice(DQ1.shape[0], n, replace=False), :]
    return X, Y
def gen_fun2(n):
    X = DP2[np.random.choice(DP2.shape[0], n, replace=False), :]
    Y = DQ2[np.random.choice(DQ2.shape[0], n, replace=False), :]
    return X, Y

def gen_fun1_t(n):
    X = DP1_t[np.random.choice(DP1_t.shape[0], n, replace=False), :]
    Y = DQ1_t[np.random.choice(DQ1_t.shape[0], n, replace=False), :]
    return X, Y

def gen_fun2_t(n):
    X = DP2_t[np.random.choice(DP2_t.shape[0], n, replace=False), :]
    Y = DQ2_t[np.random.choice(DQ2_t.shape[0], n, replace=False), :]
    return X, Y

def train_d(n, learning_rate=5e-4, N=1000, N_epoch=50, print_every=20, batch_size=32):  
    batches=n//batch_size
    assert n%batch_size==0
    print("##### Starting N_epoch=%d epochs per data trial #####"%(N_epoch))
    if True:
        X, Y = gen_fun1(n)
        total_S=[(X[i*batch_size:i*batch_size+batch_size], 
                    Y[i*batch_size:i*batch_size+batch_size]) for i in range(batches)]
        total_S=[MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in total_S]
        model_u = ConvNet_CIFAR10().cuda()
        epsilonOPT = MatConvert(np.array([-1.0]), device, dtype)
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.array([10000.0]), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.array([0.1]), device, dtype)
        sigma0OPT.requires_grad = True
        cst=MatConvert(np.ones((1,)), device, dtype) # set to 1 to meet liu etal objective
        optimizer_u = torch.optim.Adam(list(model_u.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT], lr=learning_rate)
        for t in range(N_epoch):
            for ind in range(batches):
                ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2
                S=total_S[ind]
                modelu_output = model_u(S) 
                TEMP = MMDu(modelu_output, batch_size, S, sigma, sigma0_u, ep, cst)
                mmd_val = TEMP[0]
                mmd_var = TEMP[1]
                STAT_u = crit(mmd_val, mmd_var) 
                optimizer_u.zero_grad()
                STAT_u.backward(retain_graph=True)
                optimizer_u.step()
            if t % print_every == 0:
                print('------------------------------------')
                print('Epoch:', t)
                print("mmd_value: ", mmd_val.item())
                print("Objective: ", STAT_u.item())
                print('------------------------------------')
                print('CRITERION ON NEW SET OF DATA:')            
                X1, Y1 = gen_fun2(n)
                with torch.torch.no_grad():
                    S1 = np.concatenate((X1, Y1), axis=0)
                    S1 = MatConvert(S1, device, dtype)
                    modelu_output = model_u(S1)
                    TEMP = MMDu(modelu_output, n, S1, sigma, sigma0_u, ep, cst)
                    mmd_value_temp, mmd_var_temp = TEMP[0], TEMP[1]
                    STAT_u = crit(mmd_value_temp, mmd_var_temp)
                    if True:
                        print("TEST mmd_value: ", mmd_value_temp.item())
        return model_u, torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT)), sigmaOPT ** 2, sigma0OPT ** 2, torch.tensor(X).to(device, dtype), torch.tensor(Y).to(device, dtype)

def mmdGT(X, Y, model_u, n, sigma, sigma0, device, dtype, ep):
    S = torch.cat((X, Y), dim=0)
    Fea = model_u(S)
    n = X.shape[0]
    return MMD_General(Fea, n, S, sigma, sigma0, ep, use1sample=True)
for k in range(10):
    F1='./data/result_MMDL_256_'+str(k)+'.txt'
    n=256
    fwrite('', F1, message='New File '+str(n))
    model_u, ep, sigma, sigma0, X_t, Y_t=train_d(n, learning_rate=5e-4, N=1000, N_epoch=300, print_every=20, batch_size=32)
    N=500
    m_list=list(range(4, 84, 5))
    if True:
        with torch.no_grad():
            H_u = np.zeros(N) 
            H_v = np.zeros(N)
            R_u = np.zeros(N)
            R_v = np.zeros(N)
            P_u = np.zeros(N)
            P_v = np.zeros(N)
            for i in range(len(m_list)):
                statX = []
                statY = []
                print("start testing m = %d"%m_list[i])
                m = m_list[i]
                for k in range(N):     
                    if k % 50 == 0:
                        print("start testing %d-th data trial"%k)
                    XX1, YY1 = gen_fun2_t(n+m)
                    X1     = XX1[:n]
                    Z1, Z2 = XX1[n:], YY1[n:] 
                    stat=[]
                    for j in range(100):
                            Z_temp = X1[np.random.choice(n, m, replace=False), :]
                            mmd_XZ = mmdGT(X_t, Z_temp, model_u, n, sigma, sigma0, device, dtype, ep)[0] 
                            mmd_YZ = mmdGT(Y_t, Z_temp, model_u, n, sigma, sigma0, device, dtype, ep)[0]
                            stat.append(float(mmd_XZ - mmd_YZ))
                    stat = np.sort(stat)
                    thres = stat[94]
                    Z1, Z2 = gen_fun2_t(m)
                    mmd_XZ = mmdGT(X_t, Z1, model_u, n, sigma, sigma0, device, dtype, ep)[0] 
                    mmd_YZ = mmdGT(Y_t, Z1, model_u, n, sigma, sigma0, device, dtype, ep)[0] 
                    H_u[k] = mmd_XZ - mmd_YZ < 0.0
                    R_u[k] = mmd_XZ - mmd_YZ < thres
                    P_u[k] = np.searchsorted(stat, float(mmd_XZ - mmd_YZ), side="right")/100.0
                    mmd_XZ = mmdGT(X_t, Z2, model_u, n, sigma, sigma0, device, dtype, ep)[0] 
                    mmd_YZ = mmdGT(Y_t, Z2, model_u, n, sigma, sigma0, device, dtype, ep)[0] 
                    H_v[k] = mmd_XZ - mmd_YZ > 0.0
                    R_v[k] = mmd_XZ - mmd_YZ > thres
                    P_v[k] = np.searchsorted(stat, float(mmd_XZ - mmd_YZ), side="left")/100.0
                    
                st1="n, m= "+str(n)+str('  ')+str(m)+" --- P(max|Z~X):  "+str(H_u.mean())
                st2="n, m= "+str(n)+str('  ')+str(m)+" --- P(max|Z~Y):  "+str(H_v.mean())
                st3="n, m= "+str(n)+str('  ')+str(m)+" --- P(95|Z~X):  "+str(R_u.mean())
                st4="n, m= "+str(n)+str('  ')+str(m)+" --- P(95|Z~Y):  "+str(R_v.mean())
                st5="n, m= "+str(n)+str('  ')+str(m)+" --- P(p|Z~X):  "+str(P_u.mean())
                st6="n, m= "+str(n)+str('  ')+str(m)+" --- P(p|Z~Y):  "+str(P_v.mean())
                fwrite(st1, F1)
                fwrite(st2, F1)
                fwrite(st3, F1)
                fwrite(st4, F1)
                fwrite(st5, F1)
                fwrite(st6, F1)