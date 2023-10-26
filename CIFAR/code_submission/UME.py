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
    
def crit(val, var):
    std_temp = torch.sqrt(var+10**(-9)) #this is std
    return -1 * torch.div(val, std_temp)

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



def train_d(n, V, learning_rate=5e-4, N_epoch=50, print_every=20, batch_size=32):  
    batches=n//batch_size
    assert n%batch_size==0
    print("##### Starting N_epoch=%d epochs per data trial #####"%(N_epoch))
    if True:
        X, Y = gen_fun1(n)
        total_S=[(X[i*batch_size:i*batch_size+batch_size], 
                    Y[i*batch_size:i*batch_size+batch_size]) for i in range(batches)]
        total_S=[MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in total_S] #has shape batches x 2n x d
        model_u = ConvNet_CIFAR10().cuda()
        epsilonOPT = MatConvert(np.array([-1.0]), device, dtype)
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.array([10000.0]), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.array([0.1]), device, dtype)
        sigma0OPT.requires_grad = True
        cst=MatConvert(np.ones((1,)), device, dtype) # set to 1 to meet liu etal objective
        optimizer_u = torch.optim.Adam(list(model_u.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT], lr=learning_rate)
        for _ in range(N_epoch):
            for ind in range(batches):
                ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2
                S=total_S[ind]
                modelu_output = model_u(S) 
                V_output = model_u(V)
                fea_pq = compute_feature_matrix(V_output, V, modelu_output, S, batch_size, sigma, sigma0_u, ep, cst)
                mmd_val, mmd_var = compute_UME_mean_variance(fea_pq, batch_size)
                STAT_u = crit(mmd_val, mmd_var) 
                optimizer_u.zero_grad()
                STAT_u.backward(retain_graph=True)
                optimizer_u.step()
        return model_u, torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT)), sigmaOPT ** 2, sigma0OPT ** 2, torch.tensor(X).to(device, dtype), torch.tensor(Y).to(device, dtype)

n=1920
N=300
N_f = float(N)
runs=5 #10
for _ in range(runs):
    V=gen_fun2(32)[0]
    V_t=MatConvert(V, device, dtype)
    model_u, ep, sigma, sigma0, X_t, Y_t=train_d(n, V_t, learning_rate=5e-3, N_epoch=80, print_every=16, batch_size=64)
    m_list=[96, 128, 160, 192, 216, 240, 256, 320, 384]
    with torch.no_grad():
            H_u = np.zeros(N) 
            H_v = np.zeros(N)
            R_u = np.zeros(N)
            R_v = np.zeros(N)
            P_u = np.zeros(N)
            P_v = np.zeros(N)
            print("Under this trained kernel, we run N = %d times LFI: "%N)
            Fea_X=model_u(X_t)
            Fea_Y=model_u(Y_t)
            Fea_V=model_u(V_t)
            for i in range(len(m_list)):
                statX = []
                statY = []
                print("start testing m = %d"%m_list[i])
                m = m_list[i]
                for k in range(N):     
                    stat=[]
                    for j in range(100):
                            Z_temp, _ = gen_fun2_t(m)
                            Fea_Z  = model_u(Z_temp)
                            mmd_XZ = compute_UME_mean(Fea_X, X_t, Fea_Z, Z_temp, Fea_V, V_t, sigma, sigma0, ep, cst=1.0)
                            mmd_YZ = compute_UME_mean(Fea_Y, Y_t, Fea_Z, Z_temp, Fea_V, V_t, sigma, sigma0, ep, cst=1.0)
                            stat.append(float(mmd_XZ - mmd_YZ))
                    stat = np.sort(stat)
                    thres = stat[94]
                    Z1, Z2 = gen_fun2_t(m)
                    Fea_Z1 =model_u(Z1)
                    Fea_Z2 =model_u(Z2)
                    mmd_XZ = compute_UME_mean(Fea_X, X_t, Fea_Z1, Z1, Fea_V, V_t, sigma, sigma0, ep, cst=1.0)
                    mmd_YZ = compute_UME_mean(Fea_Y, Y_t, Fea_Z1, Z1, Fea_V, V_t, sigma, sigma0, ep, cst=1.0)
                    H_u[k] = mmd_XZ - mmd_YZ < 0.0
                    R_u[k] = mmd_XZ - mmd_YZ < thres
                    P_u[k] = np.searchsorted(stat, float(mmd_XZ - mmd_YZ), side="left")/100.0
                    mmd_XZ = compute_UME_mean(Fea_X, X_t, Fea_Z2, Z2, Fea_V, V_t, sigma, sigma0, ep, cst=1.0)
                    mmd_YZ = compute_UME_mean(Fea_Y, Y_t, Fea_Z2, Z2, Fea_V, V_t, sigma, sigma0, ep, cst=1.0)
                    H_v[k] = mmd_XZ - mmd_YZ > 0.0
                    R_v[k] = mmd_XZ - mmd_YZ > thres
                    P_v[k] = np.searchsorted(stat, float(mmd_XZ - mmd_YZ), side="left")/100.0
                    
                print("n, m=",str(n)+str('  ')+str(m),"--- P(T0|Z~X): ", H_u.mean())
                print("n, m=",str(n)+str('  ')+str(m),"--- P(T0|Z~Y): ", H_v.mean())
                print("n, m=",str(n)+str('  ')+str(m),"--- P(95|Z~X): ", R_u.mean())
                print("n, m=",str(n)+str('  ')+str(m),"--- P(95|Z~Y): ", R_v.mean())
                print("n, m=",str(n)+str('  ')+str(m),"--- P(Ep|Z~X): ", P_u.mean())
                print("n, m=",str(n)+str('  ')+str(m),"--- P(Ep|Z~Y): ", P_v.mean())