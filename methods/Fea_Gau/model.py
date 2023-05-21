
from lfi.utils import *
import torch
import numpy as np


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
        output = self.model(input) + input
        return torch.zeros_like(input)

class Model(torch.nn.Module):
    def __init__(self, device, median_heuristic=False, XY_heu=None, **kwargs):
        super(Model, self).__init__()
        self.device = device
        self.model = DN().to(device)
        self.another_model = another_DN().to(device)
        self.epsilonOPT = None; #self.epsilonOPT.requires_grad = False
        self.cst = MatConvert(np.ones((1,)), device, dtype); self.cst.requires_grad = False
        if median_heuristic:
            with torch.no_grad():
                self.sigmaOPT = median(XY_heu, XY_heu)
                self.sigma0OPT = median(self.model(XY_heu), self.model(XY_heu))
        else:
            self.sigmaOPT = MatConvert(np.sqrt(np.random.rand(1)), device, dtype)
            self.sigma0OPT = MatConvert(np.sqrt(np.random.rand(1)), device, dtype) 

        self.sigmaOPT.requires_grad = False; self.sigma0OPT.requires_grad = False
        self.params = list(self.model.parameters())+[self.sigma0OPT]+[self.cst]
            
    def compute_MMD(self, XY_tr, require_grad=True, is_var_computed=True):
        batch_size = XY_tr.shape[0]//2
        ep = 0
        sigma = self.sigmaOPT ** 2; sigma0 = self.sigma0OPT ** 2
        modelu_output = self.model(XY_tr) 
        another_output =  self.another_model(XY_tr)
        mmd_val, mmd_var = MMDu(modelu_output, batch_size, another_output, sigma, sigma0, ep,    
                                    self.cst,  is_var_computed=is_var_computed)
        return mmd_val, mmd_var
    
    def compute_loss(self, XY_tr, require_grad=True, **kwargs):
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(require_grad)
        mmd_val, mmd_var = self.compute_MMD(XY_tr)
        STAT_u = mmd_val / torch.sqrt(mmd_var+10**(-8)) 
        torch.set_grad_enabled(prev)
        return -STAT_u
    
    def compute_loss_sig(self, XY_tr, require_grad=True, **kwargs):
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(require_grad)
        batch_size = XY_tr.shape[0]//2
        mmd2_val, mmd_var, K = self.compute_MMD(XY_tr, require_grad=require_grad)
        # Hij = k(Xi,Xj)-k(Xi,Yj), K=(Kx, Ky, Kxy)
        H = K[2] - K[0]
        V1 = torch.dot(H.sum(1),H.sum(1)) / batch_size**3
        V2 = (H).sum() / batch_size**2
        var_phi_X = (V1 - V2**2)
        torch.set_grad_enabled(prev)
        return - mmd2_val / torch.sqrt(var_phi_X+10**(-8)) 
    
    def compute_gram(self, X, Y, require_grad=False):
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(require_grad)
        ep = 0
        sigma = self.sigmaOPT ** 2; sigma0 = self.sigma0OPT ** 2
        Dxy = Pdist2(self.model(X), self.model(Y)); Dxy_org = Pdist2(self.another_model(X), self.another_model(Y))
        Kxy = self.cst*( (1-ep)*torch.exp(-(Dxy/sigma0)-(Dxy_org/sigma))**self.L + ep*torch.exp(-Dxy_org/sigma) )
        del Dxy, Dxy_org; gc.collect(); torch.cuda.empty_cache()
        torch.set_grad_enabled(prev)
        return Kxy
    
    def compute_scores(self, X_te, Y_te, Z_input, require_grad=False, batch_size=1024):
        # adjust batch size according to your memory capacity
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(require_grad)
        Z_input_splited = torch.split(Z_input, batch_size)
        phi_Z = torch.zeros(Z_input.shape[0]).to(X_te.device)
        for i, Z_input_batch in enumerate(Z_input_splited):
            # print(i)
            Kxz = self.compute_gram(X_te, Z_input_batch, require_grad=require_grad)
            Kyz = self.compute_gram(Y_te, Z_input_batch, require_grad=require_grad)
            phi_Z[i*batch_size: i*batch_size+Z_input_batch.shape[0]] = torch.mean(Kyz - Kxz, axis=0)
        del Kxz, Kyz, Z_input_splited; gc.collect(); torch.cuda.empty_cache()
        torch.set_grad_enabled(prev)
        return phi_Z
    
    def compute_gamma(self, X_te, Y_te, pi):
        batch_size = X_te.shape[0]
        XY_te = torch.cat((X_te, Y_te), 0)
        mmd2_val, mmd_var, K = self.compute_MMD(self, XY_te, require_grad=False, is_var_computed=False)
        Kxx = K[0]; Kyy = K[1]; Kxy = K[2]
        T_XYX = (torch.sum(Kxy)-torch.sum(Kxx)) / (batch_size**2)
        return T_XYX + pi/2*mmd2_val