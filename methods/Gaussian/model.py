
from lfi.utils import *
import torch
import numpy as np


class Const(torch.nn.Module):
    def __init__(self):
        super(Const, self).__init__()
    def forward(self, input):
        return input

class Model(torch.nn.Module):
    def __init__(self, median_heuristic=False, X_heu=None, Y_heu=None):
        super(Model, self).__init__()
        self.model = Const().to(device)
        self.another_model = Const().to(device)
        self.epsilonOPT = None; #self.epsilonOPT.requires_grad = False
        self.sigmaOPT = MatConvert(np.sqrt(np.random.rand(1)), device, dtype); self.sigmaOPT.requires_grad = True
        self.sigma0OPT = MatConvert(np.sqrt(np.random.rand(1)), device, dtype); self.sigma0OPT.requires_grad = False
        self.cst = MatConvert(np.ones((1,)), device, dtype); self.cst.requires_grad = False
        self.L = 1
        self.params = [self.sigmaOPT]
        if median_heuristic:
            self.sigmaOPT = median_heuristic(X_heu, Y_heu)
            
    def compute_MMD(self, XY_tr, require_grad=True, is_var_computed=True):
        batch_size = XY_tr.shape[0]//2
        ep = 1
        sigma = self.sigmaOPT ** 2; sigma0 = self.sigma0OPT ** 2
        modelu_output = self.model(XY_tr) 
        another_output =  self.another_model(XY_tr)
        mmd_val, mmd_var, K = MMDu(modelu_output, batch_size, another_output, sigma, sigma0, ep,    
                                    self.cst,  is_var_computed=is_var_computed)
        return mmd_val, mmd_var, K 
    
    def compute_loss(self, XY_tr, require_grad=True, **kwargs):
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(require_grad)
        mmd_val, mmd_var, K = self.compute_MMD(XY_tr)
        STAT_u = mmd_val / torch.sqrt(mmd_var+10**(-8)) 
        torch.set_grad_enabled(prev)
        return -STAT_u
    
    def compute_gram(self, X, Y, require_grad=False):
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(require_grad)
        ep = 1
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
        phi_Z = torch.zeros(Z_input.shape[0]).to(device)
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