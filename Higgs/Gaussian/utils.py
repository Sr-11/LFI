import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt
import pandas as pd
import pyroc
is_cuda = True
print('This is Gaussians utils')
class ModelLatentF(torch.nn.Module):
    """define deep networks."""
    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False

        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant

class ConstModel(torch.nn.Module):
    def __init__(self):
        super(ConstModel, self).__init__()
    def forward(self, input):
        return input

def get_item(x, is_cuda):
    """get the numpy value from a torch tensor."""
    if is_cuda:
        x = x.cpu().detach().numpy()
    else:
        x = x.detach().numpy()
    return x

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
    H = Kx+Ky-Kxy-Kxy.transpose(0,1)
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
         is_smooth=True, is_var_computed=True, use_1sample_U=True, L=1):
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
    # if is_smooth:
    #     Kx = cst*((1-epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma))**L + epsilon * torch.exp(-Dxx_org / sigma))
    #     Ky = cst*((1-epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma))**L + epsilon * torch.exp(-Dyy_org / sigma))
    #     Kxy = cst*((1-epsilon) * torch.exp(-(Dxy / sigma0) - (Dxy_org / sigma))**L + epsilon * torch.exp(-Dxy_org / sigma))
    # else:
    #     Kx = cst*torch.exp(-Dxx / sigma0)
    #     Ky = cst*torch.exp(-Dyy / sigma0)
    #     Kxy = cst*torch.exp(-Dxy / sigma0)
    Kx = torch.exp(-Dxx_org / sigma)
    Ky = torch.exp(-Dyy_org / sigma)
    Kxy = torch.exp(-Dxy_org / sigma)
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

def MMD_General(Fea, n, Fea_org, sigma, sigma0=0.1, epsilon=10 ** (-10), is_smooth=True, L=1):
    return MMDu(Fea, n, Fea_org, sigma, sigma0, epsilon, is_smooth, is_var_computed=False, use_1sample_U=False, L=L)


def MMDu_linear_kernel(Fea, len_s, is_var_computed=True, use_1sample_U=True):
    """compute value of (deep) linear-kernel MMD and std of (deep) lineaer-kernel MMD using merged data."""
    try:
        X = Fea[0:len_s, :]
        Y = Fea[len_s:, :]
    except:
        X = Fea[0:len_s].unsqueeze(1)
        Y = Fea[len_s:].unsqueeze(1)
    Kx = X.mm(X.transpose(0,1))
    Ky = Y.mm(Y.transpose(0,1))
    Kxy = X.mm(Y.transpose(0,1))
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

def load_model(model, another_model, path):
    # model = DN().cuda()
    # another_model = another_DN().cuda()
    model.load_state_dict(torch.load(path+'model.pt'))
    try:
        another_model.load_state_dict(torch.load(path+'another_model.pt'))
    except:
        another_model = ConstModel().cuda()
        print('No ResNet...')
    try:
        epsilonOPT = torch.load(path+'epsilonOPT.pt')
        sigmaOPT = torch.load(path+'sigmaOPT.pt')
        sigma0OPT = torch.load(path+'sigma0OPT.pt')
        cst = torch.load(path+'cst.pt')
    except:
        print('No eps, sigma,cst...')
    model.eval()
    another_model.eval()
    return model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst


def compute_score_func(Z, dataset_P, dataset_Q, 
                    model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, cst,
                    L=1, M=1000, 
                    dtype = torch.float, device = torch.device("cuda:0")): 
    #Z = MatConvert(Z, device, dtype)
    epsilon = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
    sigma = sigmaOPT ** 2
    sigma0 = sigma0OPT ** 2
    # print('epsilon',epsilon)
    # print('sigma',sigma)
    # print('sigma0',sigma0)
    X = dataset_P[np.random.choice(dataset_P.shape[0], M, replace=False)]
    Y = dataset_Q[np.random.choice(dataset_Q.shape[0], M, replace=False)]
    X = MatConvert(X, device, dtype)
    Y = MatConvert(Y, device, dtype)
    X_feature = model(X)
    Y_feature = model(Y)
    Z_feature = model(Z)
    X_resnet = another_model(X)
    Y_resnet = another_model(Y)
    Z_resnet = another_model(Z)
    Dxz = Pdist2(X_feature, Z_feature)
    Dyz = Pdist2(Y_feature, Z_feature)
    Dxz_org = Pdist2(X_resnet, Z_resnet)
    Dyz_org = Pdist2(Y_resnet, Z_resnet)
    Kxz = cst*((1-epsilon) * torch.exp(-(Dxz / sigma0) - (Dxz_org / sigma))**L + epsilon * torch.exp(-Dxz_org / sigma))
    Kyz = cst*((1-epsilon) * torch.exp(-(Dyz / sigma0) - (Dyz_org / sigma))**L + epsilon * torch.exp(-Dyz_org / sigma))
    phi_Z = torch.mean(Kyz - Kxz, axis=0)
    return phi_Z

def get_auc_and_x_and_y(PQhat):
    M = PQhat.shape[0]//2
    outcome = np.concatenate((np.zeros(M), np.ones(M)))
    df = pd.DataFrame(PQhat, columns=['Higgs_Default'])
    roc = pyroc.ROC(outcome, df)
    auc = roc.auc
    pred = roc.preds['Higgs_Default'] # 长度是2M
    fpr, tpr = roc._roc(pred) # False positive rate, True positive rate
    signal_to_signal_rate = tpr # 1认成1
    background_to_background_rate = 1 - fpr # 0认成0 = 1 - 0认成1
    return auc, signal_to_signal_rate, background_to_background_rate

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

def get_pval(X_score, Y_score, verbose = False):
    X_mean = np.mean(X_score)
    X_std = np.std(X_score)
    Y_mean = np.mean(Y_score)
    Y_std = np.std(Y_score)
    # 直接算平均的P的分数和方差，平均的Q的分数，然后加权
    Z_score = (10*X_mean + Y_mean)/11
    p_value = (Z_score-X_mean)/X_std
    error = ((Y_mean+Y_std)-(X_mean-X_std))/11/(X_std*(1-1/np.sqrt(len(X_score)))) - p_value
    if verbose:
        print('#datapoints =', len(X_score), ', make sure #>10000 for 2 sig digits')
        print('X_mean =', X_mean)
        print('X_std =', X_std)
        print('Y_mean =', Y_mean)
        print('Y_std =', Y_std)
        print('----------------------------------')
        print('p_value = sqrt(1100) *', p_value)
        print('error =', error)
        print('----------------------------------')
    return p_value

def early_stopping(validation_losses, epoch):
    i = np.argmin(validation_losses)
    print(i)
    if epoch - i > 10:
        return True
    else:
        return False

