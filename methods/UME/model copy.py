
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
        return output

class Model(torch.nn.Module):
    def __init__(self, device, median_heuristic_flag=True, X_heu=None, Y_heu=None):
        super(Model, self).__init__()
        # initialize parameters
        self.device = device
        self.model = DN().to(device)
        self.another_model = another_DN().to(device)
        self.epsilonOPT = MatConvert(np.zeros(1), device, dtype); self.epsilonOPT.requires_grad = True
        if median_heuristic_flag:
            with torch.no_grad():
                self.sigmaOPT = median_heuristic(self.another_model(X_heu), self.another_model(X_heu))
                self.sigma0OPT = median_heuristic(self.model(X_heu), self.model(X_heu))
                print('median_heuristic: sigma_0, sigma =', self.sigmaOPT, self.sigma0OPT)  
        else:
            self.sigmaOPT = MatConvert(np.sqrt([1.0]), device, dtype)
            self.sigma0OPT = MatConvert(np.sqrt([1.0]), device, dtype)
        self.sigmaOPT.requires_grad = True;  self.sigma0OPT.requires_grad = True
        self.cst = MatConvert(np.ones((1,)), device, dtype); self.cst.requires_grad = False
        self.L = 1
        self.params = list(self.model.parameters())+list(self.another_model.parameters())+[self.epsilonOPT]+[self.sigmaOPT]+[self.sigma0OPT]+[self.cst]
        # for storing intermediate results
        self.ep = self.sigma = self.sigma0 = 0
        self.Dxy = self.Dxy_org = self.Dxz = self.Dxz_org = self.Dyz = self.Dyz_org = torch.ones(1,1)
        self.Kxx = self.Kyy = self.Kxy = self.Kxz = self.Kyz = torch.ones(1,1)
        self.epoch = 0

    def update_tool_params(self):
        self.ep = torch.exp(self.epsilonOPT)/(1+torch.exp(self.epsilonOPT))
        self.sigma = self.sigmaOPT ** 2; self.sigma0 = self.sigma0OPT ** 2
    def update_dist_matrix(self, X, Y, Z, xy=False, xz=False, yz=False, xx=False, yy=False):
        # if not hasattr(self, 'Dxy'): self.Dxy = torch.empty((X.shape[0], Y.shape[0])).to(device)
        # if not hasattr(self, 'Dxy_org'): self.Dxy_org = torch.empty((X.shape[0], Y.shape[0])).to(device)
        # if not hasattr(self, 'Dxz'): self.Dxz = torch.empty((X.shape[0], Z.shape[0])).to(device)
        # if not hasattr(self, 'Dxz_org'): self.Dxz_org = torch.empty((X.shape[0], Z.shape[0])).to(device)
        # if not hasattr(self, 'Dyz'): self.Dyz = torch.empty((Y.shape[0], Z.shape[0])).to(device)
        # if not hasattr(self, 'Dyz_org'): self.Dyz_org = torch.empty((Y.shape[0], Z.shape[0])).to(device)
        # if not hasattr(self, 'Dxx'): self.Dxx = torch.empty((X.shape[0], X.shape[0])).to(device)
        # if not hasattr(self, 'Dxx_org'): self.Dxx_org = torch.empty((X.shape[0], X.shape[0])).to(device)
        # if not hasattr(self, 'Dyy'): self.Dyy = torch.empty((Y.shape[0], Y.shape[0])).to(device)
        # if not hasattr(self, 'Dyy_org'): self.Dyy_org = torch.empty((Y.shape[0], Y.shape[0])).to(device)
        if xy == True:
            Pdist2_(self.Dxy, self.model(X), self.model(Y))
            Pdist2_(self.Dxy_org, self.another_model(X), self.another_model(Y))
        if xz == True:
            Pdist2_(self.Dxz, self.model(X), self.model(Z))
            Pdist2_(self.Dxz_org, self.another_model(X), self.another_model(Z))
        if yz == True:
            Pdist2_(self.Dyz, self.model(Y), self.model(Z))
            Pdist2_(self.Dyz_org, self.another_model(Y), self.another_model(Z))
        if xx == True:
            Pdist2_(self.Dxx, self.model(X), self.model(X))
            Pdist2_(self.Dxx_org, self.another_model(X), self.another_model(X))
        if yy == True:
            Pdist2_(self.Dyy, self.model(Y), self.model(Y))
            Pdist2_(self.Dyy_org, self.another_model(Y), self.another_model(Y))
    def update_gram_matrix(self, X, Y, Z, xy=False, xz=False, yz=False, xx=True, yy=True):
        if xy == True:
            self.Kxy = (self.Dxy/self.sigma0).neg_().exp_()
            self.Kxy *= 1-self.ep; self.Kxy += self.ep
            self.Kxy *= (self.Dxy_org/self.sigma).neg_().exp_(); self.Kxy *= self.cst
        if xz == True:
            self.Kxz = (self.Dxz/self.sigma0).neg_().exp_()
            self.Kxz *= (1-self.ep); self.Kxz += self.ep
            self.Kxz *= (self.Dxz_org/self.sigma).neg_().exp_(); self.Kxz *= self.cst
        if yz == True:
            self.Kyz = (self.Dyz/self.sigma0).neg_().exp_()
            self.Kyz *= (1-self.ep); self.Kyz += self.ep
            self.Kyz *= (self.Dyz_org/self.sigma).neg_().exp_(); self.Kyz *= self.cst
        if xx == True:
            self.Kxx = (self.Dxx/self.sigma0).neg_().exp_()
            self.Kxx *= (1-self.ep); self.Kxx += self.ep
            self.Kxx *= (self.Dxx_org/self.sigma).neg_().exp_(); self.Kxx *= self.cst
        if yy == True:
            self.Kyy = (self.Dyy/self.sigma0).neg_().exp_()
            self.Kyy *= (1-self.ep); self.Kyy += self.ep
            self.Kyy *= (self.Dyy_org/self.sigma).neg_().exp_(); self.Kyy *= self.cst

    def compute_MMD(self, XY_tr, require_grad=True, is_var_computed=True):
        batch_size = XY_tr.shape[0]//2
        ep = torch.exp(self.epsilonOPT)/(1+torch.exp(self.epsilonOPT))
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
        prev = torch.is_grad_enabled(); torch.set_grad_enabled(require_grad)
        ep = torch.exp(self.epsilonOPT)/(1+torch.exp(self.epsilonOPT))
        sigma = self.sigmaOPT ** 2; sigma0 = self.sigma0OPT ** 2
        Dxy = Pdist2(self.model(X), self.model(Y)); Dxy_org = Pdist2(self.another_model(X), self.another_model(Y))
        Kxy = self.cst*( (1-ep)*torch.exp(-(Dxy/sigma0)-(Dxy_org/sigma))**self.L + ep*torch.exp(-Dxy_org/sigma) )
        del Dxy, Dxy_org; gc.collect(); torch.cuda.empty_cache()
        torch.set_grad_enabled(prev)
        return Kxy
    
    def compute_scores(self, X_te, Y_te, Z_input, require_grad=False, batch_size=10000):
        # 有两个参数 10000 和 100 
        # adjust batch size according to your memory capacity
        # prev = torch.is_grad_enabled(); torch.set_grad_enabled(require_grad)
        X_te_splited = torch.split(X_te, batch_size)
        Y_te_splited = torch.split(Y_te, batch_size)
        Z_input_splited = torch.split(Z_input, batch_size)
        phi_Z = torch.zeros(Z_input.shape[0]).to(self.device)
        self.update_tool_params()
        for i_Z, Z_input_batch in enumerate(Z_input_splited):
            TEM_SUM = 0
            MAX_LOOPS = 100
            COUNT_X_te = 0
            for i_X in range(min(MAX_LOOPS, len(X_te_splited))):
                self.update_dist_matrix(X_te_splited[i_X], Y_te_splited[i_X], Z_input_batch, xy=False, xz=True, yz=True)
                self.update_gram_matrix(X_te_splited[i_X], Y_te_splited[i_X], Z_input_batch, xy=False, xz=True, yz=True)
                TEM_SUM += torch.sum(self.Kyz,0) - torch.sum(self.Kxz,0)
                COUNT_X_te += X_te_splited[i_X].shape[0]
            phi_Z[i_Z*batch_size: i_Z*batch_size+Z_input_batch.shape[0]] = TEM_SUM/COUNT_X_te
        del X_te_splited, Y_te_splited, Z_input_splited; gc.collect(); torch.cuda.empty_cache()
        # torch.set_grad_enabled(prev)
        return phi_Z
    
    def compute_gamma(self, X_te, Y_te, pi, batch_size=10000):
        batch_size = 10000
        self.update_tool_params()
        nx = X_te.shape[0]
        if nx >= batch_size:
            MAX_LOOPS = 100 # 这才是MonteCarlo
        else:
            MAX_LOOPS = 1
        gamma_records = torch.zeros(MAX_LOOPS)
        for i_monte in range(MAX_LOOPS):
            if nx >= batch_size:
                idx = np.random.choice(nx, batch_size, replace=False)
                idy = np.random.choice(nx, batch_size, replace=False)
            else:
                idx = np.random.choice(nx, nx, replace=False)
                idy = np.random.choice(nx, nx, replace=False)
            X_te_batch = X_te[idx]
            Y_te_batch = Y_te[idy]
            self.update_dist_matrix(X_te_batch, Y_te_batch, None, xy=True, xz=False, yz=False, xx=True, yy=True)
            self.update_gram_matrix(X_te_batch, Y_te_batch, None, xy=True, xz=False, yz=False, xx=True, yy=True)
            EKxx = (torch.sum(self.Kxx) - torch.sum(torch.diag(self.Kxx)))/ (batch_size * (batch_size - 1))
            EKyy = (torch.sum(self.Kyy) - torch.sum(torch.diag(self.Kyy)))/ (batch_size * (batch_size - 1))
            EKxy = torch.sum(self.Kxy) / (batch_size * batch_size)
            gamma_records[i_monte] = EKxx*(pi/2-1) + EKxy*(1-pi) + EKyy*(pi/2)
        print('gamma MonteCarlo std/mean', torch.std(gamma_records)/torch.mean(gamma_records)/np.sqrt(MAX_LOOPS))
        gamma = torch.mean(gamma_records)
        return gamma
