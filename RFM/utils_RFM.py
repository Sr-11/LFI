import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt
import pandas as pd
import pyroc
is_cuda = True
import scipy
import gc
import os
from tqdm import trange
import hickle
device = torch.device("cuda:0")
dtype = torch.float32

class ConstModel(torch.nn.Module):
    def __init__(self):
        super(ConstModel, self).__init__()
    def forward(self, input):
        return input

# define network
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
        )
    def forward(self, input):
        output = self.model(input)
        return output

class another_DN(torch.nn.Module):
    def __init__(self, H=300, out=100):
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
            torch.nn.Linear(H, 28, bias=True),
        )
    def forward(self, input):
        output = input + self.model(input) 
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

# Since we use pytorch rather than autograd.np, we don't follow the inheritance kernel class in the kmod package
class NeuralKernel():
    """
    A neural net + a isotropic Gaussian kernel.
    """
    def __init__(self, model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, eps, cst):
        self.model = model
        self.another_model = another_model
        self.epsilonOPT = epsilonOPT
        self.sigmaOPT = sigmaOPT
        self.sigma0OPT = sigma0OPT
        self.eps = eps
        self.cst = cst
        self.params = list(model.parameters())+list(another_model.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT]+[eps]+[cst]

    def compute_feature_matrix(self, XY, V): 
        """
        Compute fea_pq = psi_p(V)-psi_q(V), whose shape is n x J

        Parameters
        ----------
        XY : (n1+n2) x d numpy array
        V : J x d numpy array

        Return
        ------
        fea_pq : n x J numpy array
        """
        n = len_X = XY.shape[0]//2
        J = V.shape[0]

        cst = self.cst
        sigma0 = self.sigma0OPT ** 2
        sigma = self.sigmaOPT ** 2
        epsilon = torch.exp(self.epsilonOPT)/(1+torch.exp(self.epsilonOPT))

        model_XY = self.model(XY)
        another_model_XY = self.another_model(XY)
        model_X = model_XY[0:len_X, :]
        model_Y = model_XY[len_X:, :]
        another_model_X = another_model_XY[0:len_X, :]
        another_model_Y = another_model_XY[len_X:, :]

        model_V = self.model(V)
        another_model_V = self.another_model(V)

        Dxv = Pdist2(model_X, model_V)
        Dyv = Pdist2(model_Y, model_V)
        Dxv_org = Pdist2(another_model_X, another_model_V)
        Dyv_org = Pdist2(another_model_Y, another_model_V)

        Kxv = cst* (((1-epsilon) * torch.exp(- Dxv / sigma0) + epsilon) * torch.exp(-Dxv_org / sigma))
        Kyv = cst* (((1-epsilon) * torch.exp(- Dyv / sigma0) + epsilon) * torch.exp(-Dyv_org / sigma))

        fea_pq = 1/np.sqrt(J) * (Kxv - Kyv)

        return fea_pq
        
    def compute_UME_mean_variance(self, XY, V): # compute mean and var of UME(X,Y)
        """
        Return the mean and variance of the reduced
        test statistic = \sqrt{n} UME(P, Q)^2
        The estimator of the mean is unbiased (can be negative).

        returns: (mean, variance)
        """
        # get the feature matrices psi (correlated)
        # fea_pq = psi_p(V)-psi_q(V) = n x J,
        J = V.shape[0]
        n = XY.shape[0]//2

        fea_pq = self.compute_feature_matrix(XY, V) # n x J
        
        # compute the mean 
        t1 = torch.sum(torch.mean(fea_pq, axis=0)**2) * (n/float(n-1))
        t2 = torch.mean(torch.sum(fea_pq**2, axis=1)) / float(n-1)
        UME_mean = t1 - t2

        # compute the variance
        mu = torch.mean(fea_pq, axis=0, keepdim=True) # J*1 vector
        mu = mu.t()
        # ! note that torch.dot does not support broadcasting
        UME_variance = 4.0*torch.mean(torch.matmul(fea_pq, mu)**2) - 4.0*torch.sum(mu**2)**2

        return UME_mean, UME_variance
    
    def clamp(self):
        with torch.no_grad():
            self.cst.clamp_(min=0.5, max=2.0)
            self.epsilonOPT.clamp_(min=-10.0, max=10.0)
            self.sigmaOPT.clamp_(min=0.0, max=30.0)
            self.sigma0OPT.clamp_(min=0.0, max=30.0)

    def compute_gram_matrix(self, X, Y): 
        """
        Parameters
        ----------
        XY : (n1+n2) x d numpy array

        Return
        ------
        fea_pq : n1 x n2 numpy array
        """
        with torch.no_grad():
            cst = self.cst
            sigma0 = self.sigma0OPT ** 2
            sigma = self.sigmaOPT ** 2
            epsilon = torch.exp(self.epsilonOPT)/(1+torch.exp(self.epsilonOPT))

            model_X = self.model(X)
            model_Y = self.model(Y)
            another_model_X = self.another_model(X)
            another_model_Y = self.another_model(Y)

            Dxy = Pdist2(model_X, model_Y)
            Dxy_org = Pdist2(another_model_X, another_model_Y)
            Kxy = cst* (((1-epsilon) * torch.exp(- Dxy / sigma0) + epsilon) * torch.exp(-Dxy_org / sigma))
            return Kxy

    def compute_UME_mean(self, X, Y, V): # compute mean of UME(X,Y)
        """
        Return the mean of the reduced
        Here we don't assume X.shape == Y.shape !
        X.shape = n1 x d, Y.shape = n2 x d, V.shape = J x d
        Return a scalar
        !!!! current version is biased
        """
        with torch.no_grad():
            Kxv = self.compute_gram_matrix(X,V) # n1 x J
            Kyv = self.compute_gram_matrix(Y,V) # n2 x J
            mu_p_V = torch.mean(Kxv, axis=0) # J vector
            mu_q_V = torch.mean(Kyv, axis=0) # J vector
            t1 = torch.mean((mu_p_V-mu_q_V)**2)
            t2 = 0
            UME_mean = t1 - t2
            return UME_mean
    
    def compute_scores(self, X, Y, Z, V):
        """
        T = UME^2(Z,X) - UME^2(Z,Y) = \sum f(Zi) - gamma(X,Y)
        X : n x d, Y : n x d, Z : m x d, V : J x d
        Return : [f(Zi)], a length=m vector
        """
        with torch.no_grad():
            fea_pq = self.compute_feature_matrix(torch.cat((X,Y), dim=0), V) # n x J
            gram_zw = self.compute_gram_matrix(Z, V) # m x J
            J = V.shape[0]
            # print(torch.matmul(fea_pq, gram_zw.t()))
            result = - 2/np.sqrt(J) * torch.mean(torch.matmul(fea_pq, gram_zw.t()), dim=0) # n x m
            return result
        

# save ckeckpoint
def save_model(V, kernel, epoch, folder_path):
    path = folder_path+str(epoch)+'/'
    try:
        os.makedirs(path) 
    except:
        pass
    torch.save(kernel.model.state_dict(), path+'model.pt')
    torch.save(kernel.another_model.state_dict(), path+'another_model.pt')
    torch.save(kernel.epsilonOPT, path+'epsilonOPT.pt')
    torch.save(kernel.eps, path+'eps.pt')
    torch.save(kernel.sigmaOPT, path+'sigmaOPT.pt')
    torch.save(kernel.sigma0OPT, path+'sigma0OPT.pt')
    torch.save(kernel.cst, path+'cst.pt')
    torch.save(V, path+'V.pt')

# load checkpoint
def load_model(folder_path, epoch=0):
    # print('loading model from epoch', epoch)
    with open(folder_path+'/M.h', 'rb') as file:
        M = hickle.load(file); 
    return M

def get_pval_from_evaluated_scores(X_score, Y_score, norm_or_binom=True, thres=None, verbose = False): # thres过的
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

def get_auc_from_evaluated_scores(X,Y):
    M = X.shape[0]
    outcome = np.concatenate((np.zeros(M), np.ones(M)))
    df = pd.DataFrame(torch.cat((X,Y), dim=0), columns=['Higgs_Default'])
    roc = pyroc.ROC(outcome, df)
    auc = roc.auc
    pred = roc.preds['Higgs_Default'] # 长度是2M
    #print(pred.shape)
    fpr, tpr = roc._roc(pred) # False positive rate, True positive rate
    #print(fpr.shape, tpr.shape)
    signal_to_signal_rate = tpr # 1认成1
    background_to_background_rate = 1 - fpr # 0认成0 = 1 - 0认成1
    return auc, signal_to_signal_rate, background_to_background_rate

def get_thres_from_evaluated_scores(X, Y):
    auc, x, y = get_auc_from_evaluated_scores(X,Y)
    E = 100*x+1000*(1-y)
    p_val = scipy.stats.binom.cdf(E, 1100, 1-y)
    p_list = scipy.stats.norm.ppf(p_val)
    p_list[p_list==np.inf] = 0
    #p_list = p_list[p_list.shape[0]//10 : p_list.shape[0]//10*9]
    sorted = np.sort(np.unique(torch.cat((X,Y), dim=0)), axis=None)
    i = np.argmax(p_list)

    # print(sorted[i], np.max(PQhat), np.min(PQhat))
    # print('thres=', sorted[i], ',max=', np.max(PQhat), ',min=', np.min(PQhat))
    # plt.plot(sorted, p_list)
    # plt.axvline(x=sorted[i], color='r', label='thres')
    # plt.savefig('p-thres.png')
    # print('In get_thres(), p-thres.png saved')
    # plt.show()

    return sorted[i], x[i], y[i]

##############################################################################################################

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


def get_thres_at_once(X_eval, Y_eval,
                      model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,
                      XY_sub_size = 10000, XY_MonteCarlo = 100, batch_size = 10000):
    # 用X，Y做n_eval, [X,Y]做phi(Z), 求ROC
    assert X_eval.shape[0] == Y_eval.shape[0]
    XY_test = torch.concatenate((X_eval, Y_eval), axis=0)
    n_test = XY_test.shape[0]
    Scores = torch.zeros(n_test)
    batch_size = batch_size
    for i in range(1+(n_test-1)//batch_size):
        Scores[i*batch_size : (i+1)*batch_size] =  compute_score_func(XY_test[i*batch_size : (i+1)*batch_size], X_eval, Y_eval,
                                                        model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, cst) 
    Scores = Scores.cpu().detach().numpy()
    thres, sig_to_sig, back_to_back = get_thres(Scores)
    return thres, sig_to_sig, back_to_back


###########
def get_pval_at_once(X_eval, Y_eval, X_test, Y_test,
                      model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,
                      batch_size = 10000,
                      norm_or_binom=True):
    thres,_,_ = get_thres_at_once(X_eval, Y_eval,
                        model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst)
    n_test = X_test.shape[0]
    batch_size = batch_size
    XY_test = torch.cat((X_test, Y_test), axis=0)
    Scores = torch.zeros(2*n_test)
    for i in range(1+(2*n_test-1)//batch_size):
        Scores[i*batch_size : (i+1)*batch_size] =  compute_score_func(XY_test[i*batch_size : (i+1)*batch_size], X_eval, Y_eval,
                                                        model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, cst) 
    X_scores = Scores[:n_test]
    Y_scores = Scores[n_test:]    
    del Scores
    gc.collect()
    plt.hist(X_scores.cpu().detach().numpy(), bins=100, alpha=0.5, label='X')
    plt.hist(Y_scores.cpu().detach().numpy(), bins=100, alpha=0.5, label='Y')
    plt.axvline(x=thres, color='r', linestyle='--')
    plt.legend()
    plt.show()

    pval = get_pval(X_scores, Y_scores, thres = thres, norm_or_binom=norm_or_binom)
    return pval
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