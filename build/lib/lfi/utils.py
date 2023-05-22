import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt
import pandas as pd
import pyroc
import scipy
import gc, os
from tqdm import trange

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
dtype = torch.float32
torch.backends.cudnn.deterministic = True
np.random.seed(42)
torch.manual_seed(42)

# initialize the process group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
def cleanup():
    torch.distributed.destroy_process_group()

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
    return Pdist

def Pdist2_(D, x, y):
    """in-place version of Pdist2."""
    if D.shape == (x.shape[0], y.shape[0]):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        torch.mm(x, torch.transpose(y, 0, 1), out=D)
        D *= -2.0; D += x_norm; D += y_norm
        D[D<0]=0
        return D
    else:
        D = Pdist2(x, y)
        return D

def median(X,Y,L=1):
    '''Implementation of the median heuristic. See Gretton 2012'''
    n1, d1 = X.shape
    n2, d2 = Y.shape
    assert d1 == d2, 'Dimensions of input vectors must match'
    Dxy = Pdist2(X, Y)
    mdist2 = torch.median(Dxy)
    sigma = torch.sqrt(mdist2)
    return sigma

def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True, is_unbiased = True, use_2nd = False):
    """compute value of MMD and std of MMD using kernel matrix."""
    """Kx: (n_x,n_x); Kx: (n_y,n_y); Kxy: (n_x,n_y)"""
    """Notice: their estimator is also biased, including 2nd order term (but the value is incorrect)"""
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

def get_pval_from_evaluated_scores(X_score, Y_score, pi=1/11, m=1100, thres=None, verbose = False): 
    """compute p-value from evaluated witness scores."""
    if thres == None:
        X_mean = torch.mean(X_score, dtype=dtype)
        X_std = torch.std(X_score)
        Y_mean = torch.mean(Y_score, dtype=dtype)
        Y_std = torch.std(Y_score)
        # 直接算平均的P的分数和方差，平均的Q的分数，然后加权
        Z_score = (1-pi)*X_mean + pi*Y_mean
        p_value = (Z_score-X_mean)/X_std*np.sqrt(m)
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
        signal_efficiency = torch.mean(Y_score>thres, dtype=dtype).item() # sig->sig
        background_rejection = torch.mean(X_score<thres, dtype=dtype).item() # bkg->bkg
        expected_sum = m*pi*signal_efficiency + m*(1-pi)*(1-background_rejection)
        p_val = scipy.stats.binom.cdf(expected_sum, m, 1-background_rejection)
        p_val = scipy.stats.norm.ppf(p_val)
        return p_val 

def get_auc_from_evaluated_scores(X_scores, Y_scores, plot_ROC_path=None, pyroc=False):
    """compute ROC from evaluated witness scores."""
    if pyroc:
        M = X_scores.shape[0]
        outcome = np.concatenate((np.zeros(M), np.ones(M)))
        df = pd.DataFrame(np.concatenate([X_scores,Y_scores]) , columns=['Higgs_Default'])
        roc = pyroc.ROC(outcome, df)
        auc = roc.auc
        pred = roc.preds['Higgs_Default'] # 长度是2M
        fpr, tpr = roc._roc(pred) # False positive rate, True positive rate
        signal_to_signal_rate = tpr # 1认成1
        background_to_background_rate = 1 - fpr # 0认成0 = 1 - 0认成1
        if plot_ROC_path != None:
            fig, ax = roc.plot()
            fig.savefig(plot_ROC_path)
            plt.close(fig)
    else:
        sorted = np.sort(np.unique(np.concatenate((X_scores,Y_scores), axis=0)), axis=None)
        signal_to_signal_rate = np.zeros(len(sorted))
        background_to_background_rate = np.zeros(len(sorted))
        for i in range(len(sorted)):
            signal_to_signal_rate[i] = np.mean(Y_scores>sorted[i])
            background_to_background_rate[i] = np.mean(X_scores<sorted[i])
        auc = None
    return auc, signal_to_signal_rate, background_to_background_rate
def get_thres_from_evaluated_scores(X_scores, Y_scores, pi=1/11, m=1100, plot_ROC_path=None):
    """compute t_opt from evaluated witness scores."""
    if type(X_scores) == torch.Tensor:
        X_scores = X_scores.cpu().numpy()
    if type(Y_scores) == torch.Tensor:
        Y_scores = Y_scores.cpu().numpy()
    auc, x, y = get_auc_from_evaluated_scores(X_scores,Y_scores,plot_ROC_path)
    expected_sum = m*pi*x + m*(1-pi)*(1-y)
    p_val_list = scipy.stats.norm.ppf(scipy.stats.binom.cdf(expected_sum, m, 1-y))
    p_val_list[p_val_list==np.inf] = 0
    #p_list = p_list[p_list.shape[0]//10 : p_list.shape[0]//10*9]
    sorted = np.sort(np.unique(np.concatenate((X_scores,Y_scores), axis=0)), axis=None)
    i = np.argmax(p_val_list)
    return sorted[i], sorted, p_val_list


def get_error_from_evaluated_scores(X_score, Y_score, pi, gamma, m, verbose = False):
    """compute type 1 and type 2 error from evaluated witness scores."""
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

def plot_hist(P_scores, Q_scores, path, title='', pi=None, gamma=None, thres=None, verbose=False, close=True):
    """plot histogram of scores"""
    if verbose:
        print("plotting histogram to %s"%(path))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    # if torch, to numpy
    if type(P_scores) == torch.Tensor:
        P_scores = P_scores.cpu().numpy()
    if type(Q_scores) == torch.Tensor:
        Q_scores = Q_scores.cpu().numpy()
    fig = plt.figure()
    plt.hist(P_scores, bins=50, label='$\mathrm{wit}_{Q,P}(X)$', alpha=0.5, color='r')
    plt.hist(Q_scores, bins=50, label='$\mathrm{wit}_{Q,P}(Y)$', alpha=0.5, color='b')
    P_mean_score = np.mean(P_scores)
    Q_mean_score = np.mean(Q_scores)
    plt.axvline(P_mean_score, color='r', linestyle='-', label='$\mathbb{E}\mathrm{wit}_{Q,P}(X)$')
    plt.axvline(Q_mean_score, color='b', linestyle='-', label='$\mathbb{E}\mathrm{wit}_{Q,P}(Y)$')
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
        plt.axvline(pi*Q_mean_score+(1-pi)*P_mean_score, color='y', label='$\pi$=%.2f mixed mean'%pi)
    if thres is not None:
        plt.axvline(thres, color='k', linestyle='--', label='threshold $t$')
    plt.legend()
    plt.savefig(path)
    if close:
        plt.clf()
        plt.close()
        plt.close('all')
        gc.collect()
    return fig

def plot_pval_thres(fig_path, pack1, pack2):
    t_opt, thres_opt_list, pval_opt_list = pack1
    t_cal, thres_cal_list, pval_cal_list = pack2
    fig, ax = plt.subplots()
    ax.plot(thres_opt_list, pval_opt_list, label='opt', alpha=0.5)
    ax.plot(thres_cal_list, pval_cal_list, label='cal', alpha=0.5)
    ax.axvline(x=t_opt, color='r', label='$t_{opt}$')
    ax.axhline(y=5, color='r')
    ax.legend()
    fig.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(which='minor')
    fig.savefig(fig_path)
    plt.close('all')
    return fig