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

def get_error_from_evaluated_scores(X_scores, Y_scores, 
                                    m=0, pi=0, gamma=0, 
                                    soft_or_hard=True, thres=None, verbose = False):
    X_mean = torch.mean(X_scores)
    X_std = torch.std(X_scores)
    Y_mean = torch.mean(Y_scores)
    Y_std = torch.std(Y_scores)

    mean1 = X_mean
    std1 = X_std
    type_1_error = 1-scipy.stats.norm.cdf((gamma-mean1)/std1*np.sqrt(m))

    mean2 = Y_mean*pi + X_mean*(1-pi)
    std2 = np.sqrt(pi*Y_std**2 + (1-pi)*X_std**2 + pi*(1-pi)*(X_mean-Y_mean)**2)
    type_2_error = scipy.stats.norm.cdf((gamma-mean2)/std2*np.sqrt(m))

    return type_1_error, type_2_error
    # def type_1_error_H0(self, pi, m, use_gaussian, MonteCarlo):
    #     P_scores = self.P_scores
    #     Q_scores = self.Q_scores
    #     mean = self.P_mean
    #     std = self.P_std
    #     P_mean = self.P_mean
    #     P_std = self.P_std
    #     Q_mean = self.Q_mean
    #     gamma = self.EKxx*(pi/2-1) + self.EKxy*(1-pi) + self.EKyy*(pi/2)
    #     #gamma = (pi/2)*Q_mean + (1-pi/2)*P_mean
    #     self.gamma = gamma
    #     if m==1:
    #         type_1_error = np.mean(P_scores > gamma)
    #         self.type_1_error = type_1_error
    #         return type_1_error
    #     if use_gaussian:
    #         type_1_error = 1-scipy.stats.norm.cdf((gamma-mean)/std*np.sqrt(m))
    #     else:
    #         MonteCarlo_list = np.zeros(MonteCarlo)
    #         for i in range(MonteCarlo):
    #             idx = np.random.choice(P_scores.shape[0], m, replace=False)
    #             MonteCarlo_list[i] = np.mean(P_scores[idx])
    #         type_1_error = np.mean(MonteCarlo_list > gamma)
    #         del MonteCarlo_list
    #         gc.collect()
    #     self.type_1_error = type_1_error
    #     return type_1_error
    # return 0