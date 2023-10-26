import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt

is_cuda = True

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

    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)
    if varEst == 0.0:
        print('error!!'+str(V1))
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
        Kx = torch.exp(-Dxx_org / sigma)
        Ky = torch.exp(-Dyy_org / sigma)
        Kxy = torch.exp(-Dxx_org / sigma)

    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

def MMDs(Fea, batch_n, Fea_org, sigma, cst, is_var_computed=True, use_1sample_U=True):
    X=Fea[0:batch_n, :] #has shape batch_n x out
    Y=Fea[batch_n:] #has shape batch_n x out
    Dxx = Pdist2(X, X) #has shape batch_n x batch_n
    Dyy = Pdist2(Y, Y) 
    Dxy = Pdist2(X, Y)
    Kx = cst * torch.exp(-Dxx / sigma) #has shape batch_n x batch_n
    Ky = cst * torch.exp(-Dyy / sigma)
    Kxy = cst * torch.exp(-Dxy / sigma) #has shape batch_n x batch_n
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

def MMD_General(Fea, n, Fea_org, sigma, sigma0=0.1, epsilon=10 ** (-10),use1sample=False, is_smooth=True):
    return MMDu(Fea, n, Fea_org, sigma, sigma0, epsilon, is_smooth, is_var_computed=False, use_1sample_U=use1sample)

def MMD_LFI(Fea, n, Fea_org, sigma, sigma0=0.1, epsilon=10 ** (-10), cst = 1.0, is_smooth=True, one_sample=False):
    X = Fea[0:n, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[n:2*n, :] # fetch the sample 2 (features of deep networks)
    Z = Fea[2*n:, :] # fetch the sample 3 (features of deep networks)
    X_org = Fea_org[0:n, :] # fetch the original sample 1
    Y_org = Fea_org[n:2*n, :] # fetch the original sample 2
    Z_org = Fea_org[2*n:, :] # fetch the original sample 3
    L = 1 # generalized Gaussian (if L>1)
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxz = Pdist2(X, Z)
    Dyz = Pdist2(Y, Z)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxz_org = Pdist2(X_org, Z_org)
    Dyz_org = Pdist2(Y_org, Z_org)
    if is_smooth:
        Kx = cst*((1-epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma))**L + epsilon * torch.exp(-Dxx_org / sigma))
        Ky = cst*((1-epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma))**L + epsilon * torch.exp(-Dyy_org / sigma))
        Kxz = cst*((1-epsilon) * torch.exp(-(Dxz / sigma0) - (Dxz_org / sigma))**L + epsilon * torch.exp(-Dxz_org / sigma))
        Kyx = cst*((1-epsilon) * torch.exp(-(Dyz / sigma0) - (Dyz_org / sigma))**L + epsilon * torch.exp(-Dyz_org / sigma))
    else:
        Kx = cst*torch.exp(-Dxx / sigma0)
        Ky = cst*torch.exp(-Dyy / sigma0)
    return MMD_LFI_SQUARE(Kx, Ky, Kxz, Kyx, n, len(Fea)-2*n, one_sample_U=one_sample)


def MMD_LFI_SQUARE(Kx, Ky, Kyz, Kxz, batch_n, batch_m, one_sample_U=False):
    nx = batch_n
    nz = batch_m
    if one_sample_U:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (nx * (nx - 1)))
        # one-sample U-statistic.
        xz = torch.div((torch.sum(Kxz) - torch.sum(torch.diag(Kxz))), (nx * (nz - 1)))
        yz = torch.div((torch.sum(Kyz) - torch.sum(torch.diag(Kyz))), (nx * (nz - 1)))
        mmd2 = xx - yy + 2* xz - 2* yz
    else:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (nx * (nx - 1)))
        xz = torch.div((torch.sum(Kxz)), (nx * nz))
        yz = torch.div((torch.sum(Kyz)), (nx * nz))
        mmd2 = xx - yy + 2* xz - 2* yz
    return mmd2

def fwrite(line_, file_, message='New File'):
    if line_ == '':
        with open(file_, 'w') as f:
            f.write(message)
            f.write('\n')
    else:
        with open(file_, 'a') as f:
            f.write(line_)
            f.write('\n')

def compute_feature_matrix(Fea_V, V_org, Fea, Fea_org, n, sigma, sigma0=0.1, epsilon=10 ** (-10), cst = 1.0):
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
        assert Fea.shape[0] == n*2
        J = V_org.shape[0]
        model_X = Fea[0:n, :]
        model_Y = Fea[n:, :]
        another_model_X = Fea_org[0:n, :]
        another_model_Y = Fea_org[n:, :]

        model_V = Fea_V
        another_model_V = V_org

        Dxv = Pdist2(model_X, model_V)
        Dyv = Pdist2(model_Y, model_V)
        Dxv_org = Pdist2(another_model_X, another_model_V)
        Dyv_org = Pdist2(another_model_Y, another_model_V)

        Kxv = cst* (((1-epsilon) * torch.exp(- Dxv / sigma0) + epsilon) * torch.exp(-Dxv_org / sigma))
        Kyv = cst* (((1-epsilon) * torch.exp(- Dyv / sigma0) + epsilon) * torch.exp(-Dyv_org / sigma))
        fea_pq = 1/np.sqrt(J) * (Kxv - Kyv)
        return fea_pq

def compute_UME_mean_variance(fea_pq, n, verbose=False): # compute mean and var of UME(X,Y)
        """
        Return the mean and variance of the reduced
        test statistic = \sqrt{n} UME(P, Q)^2
        The estimator of the mean is unbiased (can be negative).
        returns: (mean, variance)
        """
        # fea_pq = psi_p(V)-psi_q(V) = n x J,
        # compute the mean 
        t1 = torch.sum(torch.mean(fea_pq, axis=0)**2) * (n/float(n-1))
        t2 = torch.mean(torch.sum(fea_pq**2, axis=1)) / float(n-1)
        UME_mean = t1 - t2

        # compute the variance
        mu = torch.mean(fea_pq, axis=0, keepdim=True) # J*1 vector
        mu = mu.t()
        # ! note that torch.dot does not support broadcasting
        UME_variance = 4.0*torch.mean(torch.matmul(fea_pq, mu)**2) - 4.0*torch.sum(mu**2)**2

        if verbose:
            print('mean', UME_mean.item(), 'var', UME_variance.item())
        return UME_mean, UME_variance


def MMD_LFI(X_org, Y_org, Z_org, model_u, n, sigma, sigma0, epsilon, device, dtype, one_sample=False, is_smooth=True):
    X = model_u(X_org)
    Y = model_u(Y_org)
    Z = model_u(Z_org)
    L = 1 # generalized Gaussian (if L>1)
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxz = Pdist2(X, Z)
    Dyz = Pdist2(Y, Z)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxz_org = Pdist2(X_org, Z_org)
    Dyz_org = Pdist2(Y_org, Z_org)
    Kx = ((1-epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma))**L + epsilon * torch.exp(-Dxx_org / sigma))
    Ky = ((1-epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma))**L + epsilon * torch.exp(-Dyy_org / sigma))
    Kxz =((1-epsilon) * torch.exp(-(Dxz / sigma0) - (Dxz_org / sigma))**L + epsilon * torch.exp(-Dxz_org / sigma))
    Kyx =((1-epsilon) * torch.exp(-(Dyz / sigma0) - (Dyz_org / sigma))**L + epsilon * torch.exp(-Dyz_org / sigma))
    return MMD_LFI_SQUARE(Kx, Ky, Kxz, Kyx, n, len(Z), one_sample_U=one_sample)


def MMD_LFI_SQUARE(Kx, Ky, Kyz, Kxz, batch_n, batch_m, one_sample_U=False):
    nx = batch_n
    nz = batch_m
    if one_sample_U:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (nx * (nx - 1)))
        # one-sample U-statistic.
        xz = torch.div((torch.sum(Kxz) - torch.sum(torch.diag(Kxz))), (nx * (nz - 1)))
        yz = torch.div((torch.sum(Kyz) - torch.sum(torch.diag(Kyz))), (nx * (nz - 1)))
        mmd2 = xx - yy - 2* xz + 2* yz
    else:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (nx * (nx - 1)))
        xz = torch.div((torch.sum(Kxz)), (nx * nz))
        yz = torch.div((torch.sum(Kyz)), (nx * nz))
        mmd2 = xx - yy - 2* xz + 2* yz
    return mmd2

def compute_feature_matrix(Fea_V, V_org, Fea, Fea_org, n, sigma, sigma0=0.1, epsilon=10 ** (-10), cst = 1.0):
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
        assert Fea.shape[0] == n*2
        J = V_org.shape[0]
        model_X = Fea[0:n, :]
        model_Y = Fea[n:, :]
        another_model_X = Fea_org[0:n, :]
        another_model_Y = Fea_org[n:, :]
        model_V = Fea_V
        another_model_V = V_org
        Dxv = Pdist2(model_X, model_V)
        Dyv = Pdist2(model_Y, model_V)
        Dxv_org = Pdist2(another_model_X, another_model_V)
        Dyv_org = Pdist2(another_model_Y, another_model_V)
        Kxv = cst*((1-epsilon) * torch.exp(-(Dxv / sigma0) - (Dxv_org / sigma))**2 + epsilon * torch.exp(-Dxv_org / sigma))
        Kyv = cst*((1-epsilon) * torch.exp(-(Dyv / sigma0) - (Dyv_org / sigma))**2 + epsilon * torch.exp(-Dyv_org / sigma))
        fea_pq = 1/np.sqrt(J) * (Kxv - Kyv)
        return fea_pq

def compute_UME_mean_variance(fea_pq, n, verbose=False): # compute mean and var of UME(X,Y)
        """
        Return the mean and variance of the reduced
        test statistic = \sqrt{n} UME(P, Q)^2
        The estimator of the mean is unbiased (can be negative).
        returns: (mean, variance)
        """
        # fea_pq = psi_p(V)-psi_q(V) = n x J,
        # compute the mean 
        t1 = torch.sum(torch.mean(fea_pq, axis=0)**2) * (n/float(n-1))
        t2 = torch.mean(torch.sum(fea_pq**2, axis=1)) / float(n-1)
        UME_mean = t1 - t2

        # compute the variance
        mu = torch.mean(fea_pq, axis=0, keepdim=True) # J*1 vector
        mu = mu.t()
        # ! note that torch.dot does not support broadcasting
        UME_variance = 4.0*torch.mean(torch.matmul(fea_pq, mu)**2) - 4.0*torch.sum(mu**2)**2

        if verbose:
            print('mean', UME_mean.item(), 'var', UME_variance.item())
        return UME_mean, UME_variance

def compute_gram_matrix(Fea_X, X_org, Fea_Y, Y_org, sigma, sigma0=0.1, epsilon=10 ** (-10), cst = 1.0): 
        """
        Parameters
        ----------
        XY : (n1+n2) x d numpy array
        Return
        ------
        fea_pq : n1 x n2 numpy array
        """
        Dxy = Pdist2(Fea_X, Fea_Y)
        Dxy_org = Pdist2(X_org, Y_org)
        Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0) - (Dxy_org / sigma))**1 + epsilon * torch.exp(-Dxy_org / sigma)
        return Kxy

def compute_UME_mean(Fea_X, X_org, Fea_Y, Y_org, Fea_V, V_org, sigma, sigma0=0.1, epsilon=10 ** (-10), cst = 1.0):
        """
        Return the mean of the reduced
        Here we don't assume X.shape == Y.shape !
        X.shape = n1 x d, Y.shape = n2 x d, V.shape = J x d
        Return a scalar
        !!!! current version is biased
        """
        Kxv = compute_gram_matrix(Fea_X, X_org, Fea_V, V_org, sigma, sigma0, epsilon, cst) # n1 x J
        Kyv = compute_gram_matrix(Fea_Y, Y_org, Fea_V, V_org, sigma, sigma0, epsilon, cst) # n2 x J
        mu_p_V = torch.mean(Kxv, axis=0) # J vector
        mu_q_V = torch.mean(Kyv, axis=0) # J vector
        t1 = torch.mean((mu_p_V-mu_q_V)**2)
        t2 = 0
        UME_mean = t1 - t2
        return UME_mean

def mmd(Kx, Kxy, Ky):
    assert Kxy.shape == (Kx.shape[0], Ky.shape[0])
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    xx = (np.sum(Kx) - np.sum(np.diag(Kx)))/(nx * (nx - 1))
    yy = (np.sum(Ky) - np.sum(np.diag(Ky)))/(ny * (ny - 1))
    xy = np.sum(Kxy)/(nx * ny)
    return xx - 2 * xy + yy

def ims(img):
    return np.transpose(img.reshape(3, 32, 32), (1, 2, 0))/256

def vis_check(n, gen_fun):
    x=gen_fun(n)
    for i in x[0]:
        plt.imshow(ims(i))
        plt.figure()