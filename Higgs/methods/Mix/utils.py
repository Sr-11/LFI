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
device = torch.device("cuda:0")
dtype = torch.float32

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

class another_DN(torch.nn.Module):
    def __init__(self, H=300, out=28):
        super(another_DN, self).__init__()
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
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, out, bias=True),
        )
    def forward(self, input):
        output = self.model(input) + input
        return output

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    ###Supports m times n operation where m is not n
    ###takes input with shape (n, out) and (m, out), returns output with shape (n, m)
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    # del x_norm, y_norm
    # gc.collect()
    # torch.cuda.empty_cache()
    return Pdist

def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True, use_2nd = False):
    """compute value of MMD and std of MMD using kernel matrix."""
    """Kx: (n_x,n_x)"""
    """Kx: (n_y,n_y)"""
    """Kxy: (n_x,n_y)"""
    """Notice: their estimator is also biased, including 2nd order term (but the value is incorrect)"""
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    
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
    #print(Kx.shape, Ky.shape, Kxy.shape)
    H = Kx+Ky-Kxy-Kxy.transpose(0,1)
    #print(H.shape)
    V1 = torch.dot(H.sum(1)/ny,H.sum(1)/ny) / ny
    V2 = (H).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)
    #if varEst == 0.0:
    #    print('error!!'+str(V1))
    if use_2nd:
        V3 = 0
        return mmd2, varEst, Kxyxy
    return mmd2, varEst, Kxyxy

def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon=10 ** (-10), cst = 1.0,
         is_smooth=True, is_var_computed=True, use_1sample_U=True, L=1, kwarg=None):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    L = 1 # generalized Gaussian (if L>1)
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    if is_smooth:
        Kx = cst*((1-epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma))**L + epsilon * torch.exp(-Dxx_org / sigma))
        Ky = cst*((1-epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma))**L + epsilon * torch.exp(-Dyy_org / sigma))
        Kxy = cst*((1-epsilon) * torch.exp(-(Dxy / sigma0) - (Dxy_org / sigma))**L + epsilon * torch.exp(-Dxy_org / sigma))
    else:
        Kx = cst*torch.exp(-Dxx / sigma0)
        Ky = cst*torch.exp(-Dyy / sigma0)
        Kxy = cst*torch.exp(-Dxy / sigma0)
    if kwarg == 'Gaussian':
        Kx = torch.exp(-Dxx_org / sigma)
        Ky = torch.exp(-Dyy_org / sigma)
        Kxy = torch.exp(-Dxy_org / sigma)
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)



def load_model(model, another_model, path):
    model.load_state_dict(torch.load(path+'/model.pt'))
    try:
        another_model.load_state_dict(torch.load(path+'/another_model.pt'))
    except:
        print('No ResNet...')
    try:
        epsilonOPT = torch.load(path+'/epsilonOPT.pt')
        sigmaOPT = torch.load(path+'/sigmaOPT.pt')
        sigma0OPT = torch.load(path+'/sigma0OPT.pt')
        cst = torch.load(path+'/cst.pt')
        epsilonOPT.requires_grad = False
        sigmaOPT.requires_grad = False
        sigma0OPT.requires_grad = False
        cst.requires_grad = False
    except:
        print('No eps, sigma,cst...')
    model.eval()
    another_model.eval()
    return model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst

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


def compute_score_func(Z, X, Y, 
                    model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, cst,
                    L=1, 
                    verbose = False): 
    with torch.no_grad():
    
        if verbose:
            print('epsilonOPT =', epsilonOPT)
        if epsilonOPT == 'Scheffe':
            if verbose:
                print('It is Scheffe')
            return model(Z)[:,0]
        if epsilonOPT == 'Gaussian':
            if verbose:
                print('It is Gaussian')
            sigma = sigmaOPT**2
            Dxz_org = Pdist2(X, Z)
            Dyz_org = Pdist2(Y, Z)
            Kxz = torch.exp(-Dxz_org / sigma)
            Kyz = torch.exp(-Dyz_org / sigma)
            phi_Z = torch.mean(Kyz - Kxz, axis=0)
            return phi_Z
        if epsilonOPT == 'Fea_Gau':
            if verbose:
                print('It is Fea_Gau')
            sigma0 = sigma0OPT**2
            X_feature = model(X)
            Y_feature = model(Y)
            Z_feature = model(Z)
            Dxz = Pdist2(X_feature, Z_feature)
            Dyz = Pdist2(Y_feature, Z_feature)
            Kxz = torch.exp(-Dxz / sigma0)
            Kyz = torch.exp(-Dyz / sigma0)
            phi_Z = torch.mean(Kyz - Kxz, axis=0)
            del X, Y, Z, X_feature, Y_feature, Z_feature, Dxz, Dyz, Kxz, Kyz
            torch.cuda.empty_cache()
            gc.collect()
            return phi_Z

        epsilon = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
        sigma = sigmaOPT ** 2
        sigma0 = sigma0OPT ** 2
        X_feature = model(X)
        Y_feature = model(Y)
        Z_feature = model(Z)
        Dxz = Pdist2(X_feature, Z_feature)
        Dyz = Pdist2(Y_feature, Z_feature)
        X_resnet = another_model(X)
        Y_resnet = another_model(Y)
        Z_resnet = another_model(Z)
        Dxz_org = Pdist2(X_resnet, Z_resnet)
        Dyz_org = Pdist2(Y_resnet, Z_resnet)
        Kxz = cst*((1-epsilon) * torch.exp(-(Dxz / sigma0) - (Dxz_org / sigma))**L + epsilon * torch.exp(-Dxz_org / sigma))
        Kyz = cst*((1-epsilon) * torch.exp(-(Dyz / sigma0) - (Dyz_org / sigma))**L + epsilon * torch.exp(-Dyz_org / sigma))
        phi_Z = torch.mean(Kyz - Kxz, axis=0)
        del X, Y, X_feature, Y_feature, Z_feature, X_resnet, Y_resnet, Z_resnet, Dxz, Dyz, Dxz_org, Dyz_org, Kxz, Kyz
        torch.cuda.empty_cache()
        gc.collect()
        return phi_Z

def get_auc_and_x_and_y(PQhat):
    M = PQhat.shape[0]//2
    outcome = np.concatenate((np.zeros(M), np.ones(M)))
    df = pd.DataFrame(PQhat, columns=['Higgs_Default'])
    roc = pyroc.ROC(outcome, df)
    auc = roc.auc
    pred = roc.preds['Higgs_Default'] # 长度是2M
    #print(pred.shape)
    fpr, tpr = roc._roc(pred) # False positive rate, True positive rate
    #print(fpr.shape, tpr.shape)
    signal_to_signal_rate = tpr # 1认成1
    background_to_background_rate = 1 - fpr # 0认成0 = 1 - 0认成1
    return auc, signal_to_signal_rate, background_to_background_rate

def get_thres(PQhat):
    auc, x, y = get_auc_and_x_and_y(PQhat)
    E = 100*x+1000*(1-y)
    p_val = scipy.stats.binom.cdf(E, 1100, 1-y)
    p_list = scipy.stats.norm.ppf(p_val)
    p_list[p_list==np.inf] = 0
    #p_list = p_list[p_list.shape[0]//10 : p_list.shape[0]//10*9]
    sorted = np.sort(np.unique(PQhat), axis=None)
    i = np.argmax(p_list)

    # print(sorted[i], np.max(PQhat), np.min(PQhat))
    # print('thres=', sorted[i], ',max=', np.max(PQhat), ',min=', np.min(PQhat))
    # plt.plot(sorted, p_list)
    # plt.axvline(x=sorted[i], color='r', label='thres')
    # plt.savefig('p-thres.png')
    # print('In get_thres(), p-thres.png saved')
    # plt.show()

    return sorted[i], x[i], y[i]

# def get_thres_at_once(X_eval, Y_eval,
#                       model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,
#                       XY_sub_size = 10000, XY_MonteCarlo = 100, batch_size = 10000):
#     # 用X，Y做n_eval, [X,Y]做phi(Z), 求ROC
#     assert X_eval.shape[0] == Y_eval.shape[0]
#     XY_test = torch.concatenate((X_eval, Y_eval), axis=0)
#     n_test = XY_test.shape[0]
#     Scores = torch.zeros(n_test)
#     batch_size = batch_size
#     for i in range(1+(n_test-1)//batch_size):
#         Scores[i*batch_size : (i+1)*batch_size] =  compute_score_func(XY_test[i*batch_size : (i+1)*batch_size], X_eval, Y_eval,
#                                                         model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, cst) 
#     Scores = Scores.cpu().detach().numpy()
#     thres, sig_to_sig, back_to_back = get_thres(Scores)
#     return thres, sig_to_sig, back_to_back
def get_thres_at_once(X_eval, Y_eval, X_test, Y_test, 
                      model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,
                      batch_size = 10000):
    # 用X，Y做n_eval, [X,Y]做phi(Z), 求ROC
    XY_test = torch.concatenate((X_test, Y_test), axis=0)
    n_test = XY_test.shape[0]
    Scores = torch.zeros(n_test)
    batch_size = batch_size
    for i in range(1+(n_test-1)//batch_size):
        Scores[i*batch_size : (i+1)*batch_size] =  compute_score_func(XY_test[i*batch_size : (i+1)*batch_size], X_eval, Y_eval,
                                                        model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, cst) 
    Scores = Scores.cpu().detach().numpy()
    thres, sig_to_sig, back_to_back = get_thres(Scores)
    return thres, sig_to_sig, back_to_back

def get_pval(X_score, Y_score, norm_or_binom=True, thres=None, verbose = False): # thres过的
    if norm_or_binom==True: # 高斯
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
    if norm_or_binom==False: # 二项
        a = torch.mean(Y_score>thres, dtype=dtype).item() # sig->sig
        b = torch.mean(X_score<thres, dtype=dtype).item() # bkg->bkg
        E = 100*a + 1000*(1-b)
        p_val = scipy.stats.binom.cdf(E, 1100, 1-b)
        p_val = scipy.stats.norm.ppf(p_val)
        return p_val 

# ###########
# def get_pval_at_once(X_eval, Y_eval, X_test, Y_test,
#                       model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,
#                       batch_size = 10000,
#                       norm_or_binom=True):
#     thres,_,_ = get_thres_at_once(X_eval, Y_eval,
#                         model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst)
#     n_test = X_test.shape[0]
#     batch_size = batch_size
#     XY_test = torch.cat((X_test, Y_test), axis=0)
#     Scores = torch.zeros(2*n_test)
#     for i in range(1+(2*n_test-1)//batch_size):
#         Scores[i*batch_size : (i+1)*batch_size] =  compute_score_func(XY_test[i*batch_size : (i+1)*batch_size], X_eval, Y_eval,
#                                                         model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, cst) 
#     X_scores = Scores[:n_test]
#     Y_scores = Scores[n_test:]    
#     del Scores
#     gc.collect()
#     plt.hist(X_scores.cpu().detach().numpy(), bins=100, alpha=0.5, label='X')
#     plt.hist(Y_scores.cpu().detach().numpy(), bins=100, alpha=0.5, label='Y')
#     plt.axvline(x=thres, color='r', linestyle='--')
#     plt.legend()
#     plt.show()

#     pval = get_pval(X_scores, Y_scores, thres = thres, norm_or_binom=norm_or_binom)
#     return pval


def get_pval_at_once(X_eval, Y_eval, X_eval_test, Y_eval_test, X_test, Y_test,
                      model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,
                      batch_size = 10000,
                      norm_or_binom=True):
    # a = time.time()
    thres = np.inf
    if norm_or_binom==False:
        thres,_,_ = get_thres_at_once(X_eval, Y_eval, X_test, Y_test,
                        model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst)
    n_test = X_test.shape[0]
    batch_size = batch_size
    X_scores = torch.zeros(n_test)
    Y_scores = torch.zeros(n_test)
    # b = time.time()
    for i in range(1+(n_test-1)//batch_size):
        X_scores[i*batch_size : (i+1)*batch_size] =  compute_score_func(X_test[i*batch_size : (i+1)*batch_size], X_eval, Y_eval,
                                                        model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, cst) 
        Y_scores[i*batch_size : (i+1)*batch_size] =  compute_score_func(Y_test[i*batch_size : (i+1)*batch_size], X_eval, Y_eval,
                                                        model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, cst)
    gc.collect()
    # print(torch.min(Y_scores))
    # print(torch.max(Y_scores))
    pval = get_pval(X_scores, Y_scores, thres = thres, norm_or_binom=norm_or_binom)
    # print('time for get thres:', b-a)
    # print('time for get scores:', c-b)
    # print('time for get pval:', d-c)
    # plt.hist(X_scores.cpu().detach().numpy(), bins=100, alpha=0.5, label='X')
    # plt.hist(Y_scores.cpu().detach().numpy(), bins=100, alpha=0.5, label='Y')
    # plt.axvline(x=thres, color='r', linestyle='--')
    # plt.title('p-value = '+str(pval))
    # plt.legend()
    # plt.show()

    return pval
###############
###############

def get_thres_pval(PQhat, thres, pi=1/11, m=1100):
    M = PQhat.shape[0]//2
    Phat = PQhat[:M]<thres
    Qhat = PQhat[M:]>thres
    print('test max:', np.max(PQhat), 'test min:', np.min(PQhat))
    #print(Phat,Qhat)
    a = np.mean(Qhat).item() # sig->sig
    b = np.mean(Phat).item() # bkg->bkg
    if thres == 0.5:
        print(a,b)
    E = pi*m*a+(1-pi)*m*(1-b)
    print('a:', a, ', b:', b, ', E:', E)
    p_val = scipy.stats.binom.cdf(E, m, 1-b)
    p_val = scipy.stats.norm.ppf(p_val)
    return p_val 

    

    
def early_stopping(validation_losses, epoch):
    i = np.argmin(validation_losses)
    if epoch - i >= 5:
        return True
    else:
        return False





def plot_hist(P_scores, Q_scores):
    fig = plt.figure()
    plt.hist(P_scores, bins=100, label='P_score', alpha=0.5, color='r')
    plt.hist(Q_scores, bins=100, label='Q_score', alpha=0.5, color='b')
    P_mean_score = np.mean(P_scores)
    Q_mean_score = np.mean(Q_scores)
    plt.axvline(P_mean_score, color='r', linestyle='--', label='P_mean_score')
    plt.axvline(Q_mean_score, color='b', linestyle='--', label='Q_mean_score')
    P_std_score = np.std(P_scores)
    Q_std_score = np.std(Q_scores)
    plt.axvline(P_mean_score+P_std_score, color='r', linestyle=':')
    plt.axvline(Q_mean_score+Q_std_score, color='b', linestyle=':')
    plt.axvline(P_mean_score-P_std_score, color='r', linestyle=':')
    plt.axvline(Q_mean_score-Q_std_score, color='b', linestyle=':')
    plt.legend()
    plt.savefig('./hist.png')
    print('saved hist.png...')
    return fig