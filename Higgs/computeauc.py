import numpy as np
import torch
import sys
from utils import *
from matplotlib import pyplot as plt
import torch.nn as nn
import time
import pandas as pd
import os
from numba import cuda
from tqdm import tqdm, trange
import os
from utils import Pdist2, MatConvert
import pyroc
import pandas as pd
import matplotlib as mpl

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"  
device = torch.device("cuda:0")
dtype = torch.float32

H = 300
out = 100
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


class DN(torch.nn.Module):
    def __init__(self, H, out):
        super(DN, self).__init__()
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
            torch.nn.Linear(H, 1, bias=True),
            torch.nn.Sigmoid()
        )
        torch.nn.init.normal_(self.model[0].weight,0,0.1)
        torch.nn.init.normal_(self.model[2].weight,0,0.05)
        torch.nn.init.normal_(self.model[4].weight,0,0.05)
        torch.nn.init.normal_(self.model[6].weight,0,0.05)
        torch.nn.init.normal_(self.model[8].weight,0,0.001)
    def forward(self, input):
        output = self.model(input)
        return output

def load_model(n, epoch=0, scheffe=False):
    if scheffe:
        model = DN(300,1).cuda()
        model.load_state_dict(torch.load('./Scheffe/checkpoint%d/%d/model.pt'%(n, epoch)))
        return model,None,None,None,None
    else:
        path = './checkpoint%d/'%n+str(epoch)+'/'
        model = DN(300, 100).cuda()
        model.load_state_dict(torch.load(path+'model.pt'))
        epsilonOPT = torch.load(path+'epsilonOPT.pt')
        sigmaOPT = torch.load(path+'sigmaOPT.pt')
        sigma0OPT = torch.load(path+'sigma0OPT.pt')
        cst = torch.load(path+'cst.pt')
        return model,epsilonOPT,sigmaOPT,sigma0OPT,cst

def compute_score_func(Z, dataset_P, dataset_Q, 
                    model,epsilonOPT,sigmaOPT,sigma0OPT,cst,
                    L=1, M=1000, scheffe = False): 
    if scheffe:
        return model(Z)
    #Z = MatConvert(Z, device, dtype)
    epsilon = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
    sigma = sigmaOPT ** 2
    sigma0 = sigma0OPT ** 2
    print('epsilon',epsilon)
    print('sigma',sigma)
    print('sigma0',sigma0)
    z_feature = model(Z)
    X = dataset_P[np.random.choice(dataset_P.shape[0], M, replace=False)]
    Y = dataset_Q[np.random.choice(dataset_Q.shape[0], M, replace=False)]
    X = MatConvert(X, device, dtype)
    Y = MatConvert(Y, device, dtype)
    X_feature = model(X)
    Y_feature = model(Y)
    Dxz = Pdist2(X_feature, z_feature)
    Dyz = Pdist2(Y_feature, z_feature)
    Dxz_org = Pdist2(X, Z)
    Dyz_org = Pdist2(Y, Z)
    Kxz = cst*((1-epsilon) * torch.exp(-(Dxz / sigma0) - (Dxz_org / sigma))**L + epsilon * torch.exp(-Dxz_org / sigma))
    Kyz = cst*((1-epsilon) * torch.exp(-(Dyz / sigma0) - (Dyz_org / sigma))**L + epsilon * torch.exp(-Dyz_org / sigma))
    phi_Z = torch.mean(Kxz - Kyz, axis=0)
    return phi_Z
    
def plot_roc(roc):
    # Plot ROC curve. fig, ax = roc.plot()
    fig, ax = plt.subplots(figsize=(36, 30))
    fig.tight_layout()
    # Stylying
    ax.tick_params(labelsize=60)
    # ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    # Plot diagonal line
    ax.plot([0, 1], [1, 0], color='black', lw=1, linestyle='--')
    # Set axes limits
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([-0.02, 1.05])
    # Title and labels
    ax.set_title('ROC Curve', fontsize=60)
    ax.set_xlabel('Signal efficiency', fontsize=60) #'False Positive Rate'
    ax.set_ylabel('Background rejection', fontsize=60) #'True Positive Rate'
    # Adjust figure border
    plt.gcf().subplots_adjust(top=0.97, bottom=0.06, left=0.07, right=0.98)
    # Calculate auc
    roc._calculate_auc()
    # Calculate confidence intervals
    ci = roc.ci()
    # Set default labels
    labels = roc.preds.keys()
    # Get colormap
    #viridis = plt.cm.get_cmap("viridis", len(labels))
    for i, label in enumerate(labels):
        # Get prediction for current iteration
        pred = roc.preds[label] # 长度是2M
        # Calculate FPRs and TPRs
        fpr, tpr = roc._roc(pred) # False positive rate, True positive rate
        signal_to_signal_rate = tpr # 1认成1
        background_to_background_rate = 1 - fpr # 0认成0 = 1 - 0认成1
        # Line legend
        legend = '{0}, AUC = {1:0.2f} ({2:0.2f}-{3:0.2f})'.format(
            label, roc.auc[0, i], ci[0, i], ci[1, i]
        )
        legend = '{0}, AUC = {1:0.2f} '.format(label, roc.auc[0, i])
        plt.plot(signal_to_signal_rate, 
                background_to_background_rate, 
                lw=12, 
                #color=viridis(i),
                label = legend)[0]
    # Legend stylying
    plt.legend(fontsize=60, loc=4)
    fig.savefig('./plots/roc%d_%d.png'%(n,epoch))
    plt.clf()

def plot_hist(Phat, Qhat):
    #mpl.rcParams.update(mpl.rcParamsDefault)
    plt.xlabel('ϕ(z)')
    plt.ylabel('density')
    plt.hist(Phat, bins=100, density=True, alpha=0.5, label='background')
    plt.hist(Qhat, bins=100, density=True, alpha=0.5, label='signal')
    plt.legend(loc='upper right')
    plt.title('AUC = %.2f'%(roc.auc))
    plt.savefig('./plots/hist%d_%d.png'%(n,epoch))
    plt.clf()

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


if __name__=='__main__':
    n = int(sys.argv[1])
    epoch = int(sys.argv[2])
    flag = sys.argv[3]
    if flag == 'scheffe':
        scheffe = True
        print('scheffe')
    else:
        scheffe = False
    print('Loading model...')
    model,epsilonOPT,sigmaOPT,sigma0OPT,cst = load_model(n,epoch, scheffe)
    # Determine which features were used to train the model from filename.
    print('Loading dataset...')
    dataset = np.load('HIGGS.npy')
    dataset = standardize(dataset)
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)
    # Predict.
    print('Making predictions...')
    M = 10000
    Z = MatConvert(dataset_P[np.random.choice(dataset_P.shape[0], M, replace=False)], device, dtype)
    Phat = -compute_score_func(Z, dataset_P, dataset_Q, 
                    model,epsilonOPT,sigmaOPT,sigma0OPT,cst, scheffe=scheffe)
    Z = MatConvert(dataset_Q[np.random.choice(dataset_Q.shape[0], M, replace=False)], device, dtype)
    Qhat = -compute_score_func(Z, dataset_P, dataset_Q, 
                    model,epsilonOPT,sigmaOPT,sigma0OPT,cst, scheffe=scheffe)
    if torch.mean(Phat) > torch.mean(Qhat):
        Phat, Qhat = -Phat, -Qhat
    # Phat 比 Qhat 小, background=0, signal=1
    Phat = Phat.cpu().detach().numpy()
    Qhat = Qhat.cpu().detach().numpy()
    # Compute area under the ROC curve.
    print('Computing AUC...')
    outcome = np.concatenate((np.zeros(M), np.ones(M)))
    data = np.concatenate((Phat, Qhat))
    df = pd.DataFrame(data, columns=['Higgs'])
    roc = pyroc.ROC(outcome, df)
    print('AUC = ', roc.auc)
    print('CU = ', roc.ci())

    # Plot histogram.
    plot_hist(Phat, Qhat)
    # Plot ROC curve.
    plot_roc(roc)
