import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt
import pandas as pd
import pyroc
is_cuda = True
import scipy
import gc
from tqdm import trange
import os
device = torch.device("cuda:0")
dtype = torch.float32

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    del x_norm, y_norm
    return Pdist

def Pdist2_(D, x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    torch.mm(x, torch.transpose(y, 0, 1), out=D)
    D *= -2.0
    D += x_norm
    D += y_norm
    D[D<0]=0
    del x_norm, y_norm
    return D

def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True, is_unbiased = True, use_2nd = False):
    """compute value of MMD and std of MMD using kernel matrix."""
    """Kx: (n_x,n_x); Kx: (n_y,n_y); Kxy: (n_x,n_y)"""
    """Notice: their estimator is also biased, including 2nd order term (but the value is incorrect)"""
    # Kxxy = torch.cat((Kx,Kxy),1)
    # Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    # Kxyxy = torch.cat((Kxxy,Kyxy),0)
    nx = Kx.shape[0]; ny = Ky.shape[0]; 
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None
    # H[i,j]=k(x_i,x_j)+k(y_i,y_j)-k(x_i,y_j)-k(y_i,x_j)
    H = Kx+Ky-Kxy-Kxy.transpose(0,1)
    S = H.sum(1)
    V1 = torch.dot(S,S) / ny**3
    V2 = S.sum() / nx**2
    varEst = 4*(V1 - V2**2)
    if varEst == 0.0:
       print('error!! var=0')
    if use_2nd:
        V3 = 0
    return mmd2, varEst

def MMDu(Fea, len_s, Fea_org, sigma, sigma0, epsilon, cst,
         is_smooth=True, is_var_computed=True, use_1sample_U=True, L=1, kwarg=None):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    if is_smooth:
        Kx = cst*((1-epsilon) * torch.exp(-(Dxx / sigma0)) + epsilon) * torch.exp(-Dxx_org / sigma)
        Ky = cst*((1-epsilon) * torch.exp(-(Dyy / sigma0)) + epsilon) * torch.exp(-Dyy_org / sigma)
        Kxy = cst*((1-epsilon) * torch.exp(-(Dxy / sigma0)) + epsilon) * torch.exp(-Dxy_org / sigma)
    else:
        Kx = cst*torch.exp(-Dxx / sigma0)
        Ky = cst*torch.exp(-Dyy / sigma0)
        Kxy = cst*torch.exp(-Dxy / sigma0)
    del Dxx, Dyy, Dxy, Dxx_org, Dyy_org, Dxy_org; gc.collect(); torch.cuda.empty_cache()
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)



def get_pval_from_evaluated_scores(X_score, Y_score, thres=None, verbose = False): # thres过的
    if thres == None:
        X_mean = torch.mean(X_score, dtype=dtype)
        X_std = torch.std(X_score)
        Y_mean = torch.mean(Y_score, dtype=dtype)
        Y_std = torch.std(Y_score)
        # 直接算平均的P的分数和方差，平均的Q的分数，然后加权
        Z_score = (10*X_mean + Y_mean)/11
        p_value = (Z_score-X_mean)/X_std*np.sqrt(1100)
        if verbose:
            print('#datapoints =', len(X_score), ', make sure #>10000 for 2 sig digits')
            print('X_mean =', X_mean)
            print('X_std =', X_std)
            print('Y_mean =', Y_mean)
            print('Y_std =', Y_std)
            print('----------------------------------')
            print('p_value = ', p_value)
            print('----------------------------------')
        return p_value
    else:
        a = torch.mean(Y_score>thres, dtype=dtype).item() # sig->sig
        b = torch.mean(X_score<thres, dtype=dtype).item() # bkg->bkg
        E = 100*a + 1000*(1-b)
        p_val = scipy.stats.binom.cdf(E, 1100, 1-b)
        p_val = scipy.stats.norm.ppf(p_val)
        return p_val 

def get_thres_from_evaluated_scores(X, Y):
    auc, x, y = get_auc_from_evaluated_scores(X,Y)
    E = 100*x+1000*(1-y)
    p_val = scipy.stats.binom.cdf(E, 1100, 1-y)
    p_list = scipy.stats.norm.ppf(p_val)
    p_list[p_list==np.inf] = 0
    #p_list = p_list[p_list.shape[0]//10 : p_list.shape[0]//10*9]
    sorted = np.sort(np.unique(torch.cat((X,Y), dim=0)), axis=None)
    i = np.argmax(p_list)
    return sorted[i], x[i], y[i]

def get_error_from_evaluated_scores(X_score, Y_score, pi, gamma, m, verbose = False):
    P_mean = torch.mean(X_score)
    P_std = torch.std(X_score)
    Q_mean = torch.mean(Y_score)
    Q_std = torch.std(Y_score)
    Mix_mean = Q_mean*pi + P_mean*(1-pi)
    Mix_std = torch.sqrt( pi*Q_std**2 + (1-pi)*P_std**2 + pi*(1-pi)*(P_mean-Q_mean)**2 )
    t1 = (gamma-P_mean)/P_std; t1 = t1.cpu().numpy()
    t2 = (Mix_mean-gamma)/Q_std; t2 = t2.cpu().numpy()
    type_1_error = scipy.stats.norm.cdf( -np.sqrt(m)* t1)
    type_2_error = scipy.stats.norm.cdf( -np.sqrt(m)* t2)
    return type_1_error, type_2_error

def compute_gamma(X, Y, model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, cst, 
                    dtype = torch.float, device = torch.device("cuda:0"),
                    MonteCarlo = 10000): 
    nx = X.shape[0]
    X = MatConvert(X, device, dtype)
    Y = MatConvert(Y, device, dtype)
    L = 1 # generalized Gaussian (if L>1)
    if nx<=10000:
        if epsilonOPT=='Scheffe':
            return None, None, None
        elif epsilonOPT=='Gaussian':
            sigma = sigmaOPT ** 2
            Dxx_org = Pdist2(X, X)
            Dyy_org = Pdist2(Y, Y)
            Dxy_org = Pdist2(X, Y)
            Kxx = torch.exp(-Dxx_org / sigma)
            Kyy = torch.exp(-Dyy_org / sigma)
            Kxy = torch.exp(-Dxy_org / sigma)
        elif epsilonOPT=='Fea_Gau':
            X_feature = model(X)
            Y_feature = model(Y)
            sigma0 = sigma0OPT ** 2
            Dxx = Pdist2(X_feature, X_feature)
            Dyy = Pdist2(Y_feature, Y_feature)
            Dxy = Pdist2(X_feature, Y_feature)
            Kxx = torch.exp(-Dxx / sigma0)
            Kyy = torch.exp(-Dyy / sigma0)
            Kxy = torch.exp(-Dxy / sigma0)
        else:
            X_feature = model(X)
            Y_feature = model(Y)
            X_resnet = another_model(X)
            Y_resnet = another_model(Y)
            sigma = sigmaOPT ** 2
            sigma0 = sigma0OPT ** 2
            epsilon = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
            Dxx = Pdist2(X_feature, X_feature)
            Dyy = Pdist2(Y_feature, Y_feature)
            Dxy = Pdist2(X_feature, Y_feature)
            Dxx_org = Pdist2(X_resnet, X_resnet)
            Dyy_org = Pdist2(Y_resnet, Y_resnet)
            Dxy_org = Pdist2(X_resnet, Y_resnet)
            Kxx = cst*((1-epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma))**L + epsilon * torch.exp(-Dxx_org / sigma))
            Kyy = cst*((1-epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma))**L + epsilon * torch.exp(-Dyy_org / sigma))
            Kxy = cst*((1-epsilon) * torch.exp(-(Dxy / sigma0) - (Dxy_org / sigma))**L + epsilon * torch.exp(-Dxy_org / sigma))
            del X, Y, X_feature, Y_feature, X_resnet, Y_resnet, Dxx, Dyy, Dxy, Dxx_org, Dyy_org, Dxy_org

        EKxx = (torch.sum(Kxx) - torch.sum(torch.diag(Kxx)))/ (nx * (nx - 1))
        EKyy = (torch.sum(Kyy) - torch.sum(torch.diag(Kyy)))/ (nx * (nx - 1))
        EKxy = torch.sum(Kxy) / (nx * nx)
        torch.cuda.empty_cache()
        gc.collect()
        EKxx = EKxx.cpu().detach().numpy()
        EKyy = EKyy.cpu().detach().numpy()
        EKxy = EKxy.cpu().detach().numpy()
        return EKxx, EKyy, EKxy    
    else:
        print("WARNING: Out of Memory, use MonteCarlo...")
        EKxx = np.zeros(MonteCarlo)
        EKyy = np.zeros(MonteCarlo)
        EKxy = np.zeros(MonteCarlo)
        for i in trange(MonteCarlo):
            idx = np.random.choice(nx, 10000, replace=False)
            idy = np.random.choice(nx, 10000, replace=False)
            Dxx = Pdist2(X_feature[idx], X_feature[idx])
            Dyy = Pdist2(Y_feature[idy], Y_feature[idy])
            Dxy = Pdist2(X_feature[idx], Y_feature[idy])
            Dxx_org = Pdist2(X_resnet[idx], X_resnet[idx])
            Dyy_org = Pdist2(Y_resnet[idy], Y_resnet[idy])
            Dxy_org = Pdist2(X_resnet[idx], Y_resnet[idy])
            Kxx = cst*((1-epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma))**L + epsilon * torch.exp(-Dxx_org / sigma))
            Kyy = cst*((1-epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma))**L + epsilon * torch.exp(-Dyy_org / sigma))
            Kxy = cst*((1-epsilon) * torch.exp(-(Dxy / sigma0) - (Dxy_org / sigma))**L + epsilon * torch.exp(-Dxy_org / sigma))
            EKxx[i] = (torch.sum(Kxx) - torch.sum(torch.diag(Kxx)))/ 10000 / (10000 - 1)
            EKyy[i] = (torch.sum(Kyy) - torch.sum(torch.diag(Kyy)))/ 10000 / (10000 - 1)
            EKxy[i] = torch.sum(Kxy) / 10000 / 10000
        EKxx_mean = np.mean(EKxx)
        EKyy_mean = np.mean(EKyy)
        EKxy_mean = np.mean(EKxy)
        EKxx_std = np.std(EKxx)
        EKyy_std = np.std(EKyy)
        EKxy_std = np.std(EKxy)
        print('Error =', np.sqrt(EKxx_std**2 + EKyy_std**2 + EKxy_std**2))
        del X, Y, X_feature, Y_feature, X_resnet, Y_resnet, Dxx, Dyy, Dxy, Dxx_org, Dyy_org, Dxy_org, Kxx, Kyy, Kxy, EKxx_std, EKyy_std, EKxy_std
        torch.cuda.empty_cache()
        gc.collect()
        return EKxx_mean, EKyy_mean, EKxy_mean


def get_auc_from_evaluated_scores(X, Y):
    M = X.shape[0]
    outcome = np.concatenate((np.zeros(M), np.ones(M)))
    df = pd.DataFrame(np.concatenate([X,Y]) , columns=['Higgs_Default'])
    roc = pyroc.ROC(outcome, df)
    auc = roc.auc
    pred = roc.preds['Higgs_Default'] # 长度是2M
    #print(pred.shape)
    fpr, tpr = roc._roc(pred) # False positive rate, True positive rate
    #print(fpr.shape, tpr.shape)
    signal_to_signal_rate = tpr # 1认成1
    background_to_background_rate = 1 - fpr # 0认成0 = 1 - 0认成1
    return auc, signal_to_signal_rate, background_to_background_rate


def median_heuristic(X,Y,L=1):
    '''Implementation of the median heuristic. See Gretton 2012
    '''
    n1, d1 = X.shape
    n2, d2 = Y.shape
    assert d1 == d2, 'Dimensions of input vectors must match'
    Dxy = Pdist2(X, Y)
    print(Dxy.shape)
    mdist2 = torch.median(Dxy)
    sigma = torch.sqrt(mdist2)
    return sigma

def plot_hist(P_scores, Q_scores, path, title, verbose=False, pi=None, gamma=None):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    # if torch, to numpy
    if type(P_scores) == torch.Tensor:
        P_scores = P_scores.cpu().numpy()
    if type(Q_scores) == torch.Tensor:
        Q_scores = Q_scores.cpu().numpy()
    fig = plt.figure()
    plt.hist(P_scores, bins=100, label='$\mathrm{wit}_{Q,P}(X)$', alpha=0.5, color='r')
    plt.hist(Q_scores, bins=100, label='$\mathrm{wit}_{Q,P}(Y)$', alpha=0.5, color='b')
    P_mean_score = np.mean(P_scores)
    Q_mean_score = np.mean(Q_scores)
    plt.axvline(P_mean_score, color='r', linestyle='--', label='$\mathbb{E}\mathrm{wit}_{Q,P}(X)$')
    plt.axvline(Q_mean_score, color='b', linestyle='--', label='$\mathbb{E}\mathrm{wit}_{Q,P}(Y)$')
    P_std_score = np.std(P_scores)
    Q_std_score = np.std(Q_scores)
    plt.axvline(P_mean_score+P_std_score, color='r', linestyle=':')
    plt.axvline(Q_mean_score+Q_std_score, color='b', linestyle=':')
    plt.axvline(P_mean_score-P_std_score, color='r', linestyle=':')
    plt.axvline(Q_mean_score-Q_std_score, color='b', linestyle=':')
    plt.title(title)
    plt.xlabel('witness')
    plt.ylabel('frequency')
    if gamma is not None:
        plt.axvline(gamma.cpu().numpy(), color='g', label='threshold $\gamma$')
    if pi is not None:
        plt.axvline(pi*Q_mean_score+(1-pi)*P_mean_score, color='y', label='$\pi$ mixed mean')
    plt.legend()
    plt.savefig(path)
    plt.clf()
    plt.close()
    plt.close('all')
    gc.collect()
    if verbose:
        print('saved hist.png...')
    return fig








class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = DN(out=1).to(device)
        self.params = list(self.model.parameters())
    def pval_crit(self, X_scores, Y_scores):
        X_mean = torch.mean(X_scores)
        X_std = torch.std(X_scores)
        Y_mean = torch.mean(Y_scores)
        return (Y_mean - X_mean) / X_std
    def wiki_crit(self, X_scores, Y_scores):
        Ex_g = torch.mean(X_scores)
        Ey_g = torch.mean(Y_scores)
        Ey_g2 = torch.mean(Y_scores**2)
        crit = Ex_g - Ey_g - Ey_g2/4
        return crit
    def heur_crit(self, X_scores, Y_scores):
        X_mean = torch.mean(X_scores)
        Y_mean = torch.mean(Y_scores)
        std = torch.std(Y_scores-X_scores)
        return (Y_mean - X_mean) / std
    def compute_loss(self, XY_tr, require_grad=True, method=''):
        batch_size = XY_tr.shape[0]//2
        prev = torch.is_grad_enabled(); torch.set_grad_enabled(require_grad)
        model_output = self.model(XY_tr)
        if method == 'mean/var':
            crit = self.pval_crit(model_output[:batch_size], model_output[batch_size:]) 
        elif method == 'chi_square_on_wiki':
            crit = self.wiki_crit(model_output[:batch_size], model_output[batch_size:])
        elif method == 'heuristic':
            crit = self.heur_crit(model_output[:batch_size], model_output[batch_size:])
        torch.set_grad_enabled(prev)
        return - crit
    def compute_scores(self, Z_input, require_grad=False):
        prev = torch.is_grad_enabled(); torch.set_grad_enabled(require_grad)
        model_output = self.model(Z_input)
        torch.set_grad_enabled(prev)
        return model_output
