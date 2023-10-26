
from lfi.utils import *
import torch
import numpy as np

class Model(torch.nn.Module):
    def __init__(self, device, median_heuristic=False, XY_heu=None, **kwargs):
        super(Model, self).__init__()
        self.device=device
        if median_heuristic:
            print('median_heuristic: ', end='')
            self.sigmaOPT = median(XY_heu, XY_heu)
            print(self.sigmaOPT)
        else:
            self.sigmaOPT = MatConvert(np.ones([1,1]), device, dtype)
        self.sigmaOPT.requires_grad = True
        self.params = [self.sigmaOPT]
        
    def compute_MMD(self, XY_tr, is_var_computed=True):
        batch_size = XY_tr.shape[0]//2
        ep = 1; sigma0 = 1; cst=1; sigma = self.sigmaOPT ** 2
        modelu_output = XY_tr; another_output = XY_tr
        mmd_val, mmd_var = MMDu(modelu_output, batch_size, another_output, sigma, sigma0, ep, cst,  is_var_computed=is_var_computed)
        return mmd_val, mmd_var
    def compute_loss(self, XY_tr, **kwargs):
        mmd_val, mmd_var = self.compute_MMD(XY_tr)
        STAT_u = mmd_val / torch.sqrt(mmd_var+10**(-8))
        # print(self.sigmaOPT)
        return -STAT_u
    
    def compute_gram(self, X, Y, require_grad=False):
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(require_grad)
        sigma = self.sigmaOPT ** 2
        Dxy_org = Pdist2(X, Y)
        Kxy = torch.exp(-Dxy_org/sigma)
        del Dxy_org; gc.collect(); torch.cuda.empty_cache()
        torch.set_grad_enabled(prev)
        return Kxy
    def compute_scores(self, X_te, Y_te, Z_input, require_grad=False, batch_size=1024):
        X_te = X_te[:32768]; Y_te = Y_te[:32768]
        # adjust batch size according to your memory capacity
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(require_grad)
        Z_input_splited = torch.split(Z_input, batch_size)
        phi_Z = torch.zeros(Z_input.shape[0]).to(self.device)
        for i, Z_input_batch in enumerate(Z_input_splited):
            # print(i)
            Kxz = self.compute_gram(X_te, Z_input_batch, require_grad=require_grad)
            Kyz = self.compute_gram(Y_te, Z_input_batch, require_grad=require_grad)
            phi_Z[i*batch_size: i*batch_size+Z_input_batch.shape[0]] = torch.mean(Kyz - Kxz, axis=0)
        del Kxz, Kyz, Z_input_splited; gc.collect(); torch.cuda.empty_cache()
        torch.set_grad_enabled(prev)
        return phi_Z
    
    # def compute_gamma(self, X_te, Y_te, pi):
    #     batch_size = X_te.shape[0]
    #     XY_te = torch.cat((X_te, Y_te), 0)
    #     mmd2_val, mmd_var = self.compute_MMD(self, XY_te, require_grad=False, is_var_computed=False)
    #     Kxx = self.Kxx; Kyy = self.Kyy; Kxy = self.Kxy
    #     T_XYX = (torch.sum(Kxy)-torch.sum(Kxx)) / (batch_size**2)
    #     return T_XYX + pi/2*mmd2_val