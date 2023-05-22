
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

class Model(torch.nn.Module):
    def __init__(self, device, median_heuristic=False, XY_heu=None, **kwargs):
        super(Model, self).__init__()
        self.device = device
        self.model = DN().to(device)
        if median_heuristic:
            with torch.no_grad():
                self.sigma0OPT = median(self.model(XY_heu), self.model(XY_heu))
                print('sigma = ', self.sigma0OPT)
        else:
            self.sigma0OPT = MatConvert(np.sqrt(np.random.rand(1)), device, dtype) 
        self.sigma0OPT.requires_grad = True
        self.params = list(self.model.parameters())+[self.sigma0OPT]
            
    def compute_MMD(self, XY_tr, is_var_computed=True):
        batch_size = XY_tr.shape[0]//2
        ep = 1; cst=1
        sigma0 = self.sigma0OPT ** 2
        model_output = self.model(XY_tr) 
        mmd_val, mmd_var = MMDu(model_output, batch_size, model_output, sigma0, sigma0, ep,    
                                    cst,  is_var_computed=is_var_computed)
        return mmd_val, mmd_var
    def compute_loss(self, XY_tr, **kwargs):
        mmd_val, mmd_var = self.compute_MMD(XY_tr)
        STAT_u = mmd_val / torch.sqrt(mmd_var+10**(-8)) 
        return -STAT_u
    
    def compute_gram(self, X, Y, require_grad=False):
        sigma0 = self.sigma0OPT ** 2
        Dxy = Pdist2(self.model(X), self.model(Y)); 
        Kxy = torch.exp(-(Dxy/sigma0))
        del Dxy; gc.collect(); torch.cuda.empty_cache()
        return Kxy
    
    def compute_scores(self, X_ev, Y_ev, Z_input, require_grad=False, batch_size=8192, max_loops=4):
        # adjust batch size according to your memory capacity
        X_ev_splited = torch.split(X_ev, batch_size)
        Y_ev_splited = torch.split(Y_ev, batch_size)
        Z_input_splited = torch.split(Z_input, batch_size)
        phi_Z = torch.zeros(Z_input.shape[0]).to(X_ev.device)
        for i_Z, Z_input_batch in enumerate(Z_input_splited):
            cum_sum = 0
            cum_count = 0
            for i_X in range(min(len(X_ev_splited), max_loops)):
                Kxz = self.compute_gram(X_ev_splited[i_X], Z_input_batch)
                Kyz = self.compute_gram(Y_ev_splited[i_X], Z_input_batch)
                cum_sum += torch.sum(Kyz,0) - torch.sum(Kxz,0)
                cum_count += X_ev_splited[i_X].shape[0]
            phi_Z[i_Z*batch_size: i_Z*batch_size+Z_input_batch.shape[0]] = cum_sum/cum_count
        del X_ev_splited, Y_ev_splited, Z_input_splited; gc.collect(); torch.cuda.empty_cache()
        return phi_Z
    
    # def compute_gamma(self, X_ev, Y_ev, pi):
    #     batch_size = X_ev.shape[0]
    #     XY_ev = torch.cat((X_ev, Y_ev), 0)
    #     mmd2_val, mmd_var = self.compute_MMD(self, XY_ev, require_grad=False, is_var_computed=False)
    #     Kxx = K[0]; Kyy = K[1]; Kxy = K[2]
    #     T_XYX = (torch.sum(Kxy)-torch.sum(Kxx)) / (batch_size**2)
    #     return T_XYX + pi/2*mmd2_val