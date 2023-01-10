import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt

is_cuda = True

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

def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
    """compute value of MMD and std of MMD using kernel matrix."""
    """Kx: (n_x,n_x)"""
    """Kx: (n_y,n_y)"""
    """Kxy: (n_x,n_y)"""
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
    return mmd2, varEst, Kxyxy

def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon=10 ** (-10), cst = 1.0,
         is_smooth=True, is_var_computed=True, use_1sample_U=True):
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

    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

def MMD_General(Fea, n, Fea_org, sigma, sigma0=0.1, epsilon=10 ** (-10), is_smooth=True):
    return MMDu(Fea, n, Fea_org, sigma, sigma0, epsilon, is_smooth, is_var_computed=False, use_1sample_U=False)


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


def LFI_plot(n_list, title="LFI_with_Blob" ,path='./data/', with_error_bar=True):
    Results_list = []
    fig = plt.figure(figsize=(10, 8))
    ZX_success = np.zeros(len(n_list))
    ZY_success = np.zeros(len(n_list))
    ZX_err = np.zeros(len(n_list))
    ZY_err = np.zeros(len(n_list))
    for i,n in enumerate(n_list):
        Results_list.append(np.load(path+'LFI_%d.npy'%n))
        ZX_success[i] = np.mean(Results_list[-1][0,:])
        ZY_success[i] = np.mean(Results_list[-1][1,:])
        ZX_err[i] = np.std(Results_list[-1][0,:])
        ZY_err[i] = np.std(Results_list[-1][1,:])
    if with_error_bar==True:
        plt.errorbar(n_list, ZX_success, yerr=ZX_err, label='Z~X')
        plt.errorbar(n_list, ZY_success, yerr=ZY_err, label='Z~Y')
    else:
        plt.plot(n_list, ZX_success, label='Z~X')
        plt.plot(n_list, ZY_success, label='Z~Y')
    print('Success rates:')
    print(ZX_success)
    print(ZY_success)
    plt.xlabel("n samples", fontsize=20)
    plt.ylabel("P(success)", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(title+'.png')
    plt.close()
    return fig