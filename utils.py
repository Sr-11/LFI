import numpy as np
import torch
import torch.utils.data

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
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
    yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
    if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
    else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
    mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None

    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)
    if  varEst == 0.0:
        print('error!!'+str(V1))

    return mmd2, varEst, Kxyxy

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

def MMD_LFI_STAT(Fea, Fea_org, batch_n, batch_m, sigma=0.1, cst=1.0):
    """computes the MMD squared statistics."""
    X=Fea[0:batch_n, :] #has shape batch_n x out
    Y=Fea[batch_n: 2*batch_n, :] #has shape batch_n x out
    Z=Fea[2*batch_n:, :] #has shape batch_m x out
    Dxx = Pdist2(X, X) #has shape batch_n x batch_n
    Dyy = Pdist2(Y, Y) 
    Dzz = Pdist2(Z, Z)
    Dxz = Pdist2(X, Z)
    Dyz = Pdist2(Y, Z)
    Kx = cst * torch.exp(-Dxx / sigma) #has shape batch_n x batch_n
    Ky = cst * torch.exp(-Dyy / sigma)
    Kz = cst * torch.exp(-Dzz / sigma) #has shape batch_m x batch_m
    Kxz = cst * torch.exp(-Dxz / sigma) #has shape batch_n x batch_m
    Kyz = cst * torch.exp(-Dyz / sigma)
    sq=MMD_LFI_SQUARE(Kx, Ky, Kyz, Kxz, batch_n, batch_m)
    return sq, MMD_LFI_VAR(Kx, Ky, Kz, Kyz, Kxz, batch_n, batch_m, sq)

def MMD_LFI_VAR(Kx, Ky, Kz, Kyz, Kxz, batch_n, batch_m, mean_H):
    '''computes the MMD squared variance.'''
    #One is suppose to set off some biased sample mean/variance estimate or something (i.e. /n vs /(n-1)) but I'm not sure how to do that here
    #H_{ijk}=Kx[i:j]-Ky[i:j]-2*Kyz[i:k]+2*Kxz[i:k]-mean_H and has expectation 0
    #V_mn=Expected of H_{ijk}*H_{ilk}
    #V_nn=Expected of H_{ijk}*H_{ijl}
    V_mn= 0.0
    V_nn= 0.0
    return V_mn/(batch_n*batch_m)+V_nn/(batch_n*batch_n)

def MMD_General(Fea, n, m, Fea_org, sigma=0.1, cst=1.0):
    X = Fea[0:n, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[n:, :] # fetch the sample 2 (features of deep networks)
    Dxx = Pdist2(X, X) #shape n by n
    Dyy = Pdist2(Y, Y) #shape m by m
    Dxy = Pdist2(X, Y) #shape n by m
    if True:
        Kx = cst * torch.exp(-Dxx / sigma)
        Ky = cst *torch.exp(-Dyy / sigma)
        Kxy = cst * torch.exp(-Dxy / sigma)
    return h1_mean_var_gram(Kx, Ky, Kxy, False)

def relu(tensor):
    """ReLU activation function."""
    return torch.max(tensor, torch.zeros_like(tensor)).cuda()