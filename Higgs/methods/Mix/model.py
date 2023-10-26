
from lfi.utils import MatConvert, dtype, median, Pdist2_, MMDu
import torch
import numpy as np

class DN(torch.nn.Module):
    def __init__(self, H=300, out=300):
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
    """
    The torch model for lfi, the following 4 functions must be implemented:
    0. __init__(device, median_heuristic_flag=True, X_heu=None, Y_heu=None, **kwargs):
        device is where the model should stored, e.g. torch.device('cuda:0')
        self.params must be initialized as trainable parameters
    1. compute_loss(XY_train): 
        compute the loss for the training data.
        INPUT: XY_train.shape = (2*batch_size, dim_of_data)
        OUTPUT: loss objective
    2. compute_scores(X_ev, Y_ev, Z_input):
        compute the witness scores for the test data.
        INPUT: X_ev.shape = (batch_size, dim_of_data)
               Y_ev.shape = (batch_size, dim_of_data)
               Z_input.shape = (test_size, dim_of_data)
        OUTPUT: scores.shape = (test_size,) where scores[i] = \mu_{Y_ev}(Z_input[i]) - \mu_{X_ev}(Z_input[i])
    3. compute_gamma(X_ev, Y_ev, pi): 
        compute the threshold \gamma for our test.
        INPUT: X_ev.shape = (batch_size, dim_of_data)
               Y_ev.shape = (batch_size, dim_of_data)
        OUTPUT: \gamma(X_ev, Y_ev)
    """
    def __init__(self, device, median_heuristic=True, XY_heu=None, **kwargs):
        super(Model, self).__init__()
        # initialize parameters
        self.device = device
        self.model = DN().to(device)
        self.another_model = another_DN().to(device)
        self.epsilonOPT = MatConvert(np.zeros(1), device, dtype); self.epsilonOPT.requires_grad = True
        if median_heuristic:
            with torch.no_grad():
                self.sigmaOPT = median(self.another_model(XY_heu), self.another_model(XY_heu))
                self.sigma0OPT = median(self.model(XY_heu), self.model(XY_heu))
                print('median_heuristic: sigma_0, sigma =', self.sigmaOPT, self.sigma0OPT)  
        else:
            self.sigmaOPT = MatConvert(np.sqrt([1.0]), device, dtype)
            self.sigma0OPT = MatConvert(np.sqrt([1.0]), device, dtype)
        self.sigmaOPT.requires_grad = True;  self.sigma0OPT.requires_grad = True
        self.cst = MatConvert(np.ones((1,)), device, dtype); self.cst.requires_grad = False
        self.params = list(self.model.parameters())+list(self.another_model.parameters())+[self.epsilonOPT]+[self.sigmaOPT]+[self.sigma0OPT]+[self.cst]
        # for storing intermediate results
        self.ep = 0; self.sigma = 0; self.sigma0 = 0
        self.Dxy=torch.ones(1); self.Dxy_org=torch.ones(1); self.Dxz=torch.ones(1); self.Dxz_org=torch.ones(1); self.Dyz=torch.ones(1); self.Dyz_org=torch.ones(1)
        self.Kxx=torch.ones(1); self.Kyy=torch.ones(1); self.Kxy=torch.ones(1); self.Kxz=torch.ones(1); self.Kyz=torch.ones(1)

    # compute the gram matrix. 
    def update_tool_params(self):
        self.ep = torch.exp(self.epsilonOPT)/(1+torch.exp(self.epsilonOPT))
        self.sigma = self.sigmaOPT ** 2; self.sigma0 = self.sigma0OPT ** 2
    def update_dist_matrix(self, X, Y, Z, xy=False, xz=False, yz=False, xx=False, yy=False):
        if xy == True:
            if 'Dxy' not in self.__dict__: self.Dxy = torch.ones(1,1)
            self.Dxy = Pdist2_(self.Dxy, self.model(X), self.model(Y))
            if 'Dxy_org' not in self.__dict__: self.Dxy_org = torch.ones(1,1)
            self.Dxy_org = Pdist2_(self.Dxy_org, self.another_model(X), self.another_model(Y))
        if xz == True:
            if 'Dxz' not in self.__dict__: self.Dxz = torch.ones(1,1)
            self.Dxz = Pdist2_(self.Dxz, self.model(X), self.model(Z))
            if 'Dxz_org' not in self.__dict__: self.Dxz_org = torch.ones(1,1)
            self.Dxz_org = Pdist2_(self.Dxz_org, self.another_model(X), self.another_model(Z))
        if yz == True:
            if 'Dyz' not in self.__dict__: self.Dyz = torch.ones(1,1)
            self.Dyz = Pdist2_(self.Dyz, self.model(Y), self.model(Z))
            if 'Dyz_org' not in self.__dict__: self.Dyz_org = torch.ones(1,1)
            self.Dyz_org = Pdist2_(self.Dyz_org, self.another_model(Y), self.another_model(Z))
        if xx == True:
            if 'Dxx' not in self.__dict__: self.Dxx = torch.ones(1,1)
            self.Dxx = Pdist2_(self.Dxx, self.model(X), self.model(X))
            if 'Dxx_org' not in self.__dict__: self.Dxx_org = torch.ones(1,1)
            self.Dxx_org = Pdist2_(self.Dxx_org, self.another_model(X), self.another_model(X))
        if yy == True:
            if 'Dyy' not in self.__dict__: self.Dyy = torch.ones(1,1)
            self.Dyy = Pdist2_(self.Dyy, self.model(Y), self.model(Y))
            if 'Dyy_org' not in self.__dict__: self.Dyy_org = torch.ones(1,1)
            self.Dyy_org = Pdist2_(self.Dyy_org, self.another_model(Y), self.another_model(Y))
    def update_gram_matrix(self, X, Y, Z, xy=False, xz=False, yz=False, xx=False, yy=False):
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

    # def compute_MMD(self, XY_tr, is_var_computed=True):
    #     batch_size = XY_tr.shape[0]//2
    #     ep = torch.exp(self.epsilonOPT)/(1+torch.exp(self.epsilonOPT))
    #     sigma = self.sigmaOPT ** 2; sigma0 = self.sigma0OPT ** 2
    #     modelu_output = self.model(XY_tr) 
    #     another_output =  self.another_model(XY_tr)
    #     mmd_val, mmd_var = MMDu(modelu_output, batch_size, another_output, sigma, sigma0, ep,    
    #                                 self.cst,  is_var_computed=is_var_computed)
    #     return mmd_val, mmd_var
    def compute_loss(self, XY_tr):
        batch_size = XY_tr.shape[0]//2
        ep = torch.exp(self.epsilonOPT)/(1+torch.exp(self.epsilonOPT))
        sigma = self.sigmaOPT ** 2; sigma0 = self.sigma0OPT ** 2
        modelu_output = self.model(XY_tr) 
        another_output =  self.another_model(XY_tr)
        mmd_val, mmd_var = MMDu(modelu_output, batch_size, another_output, sigma, sigma0, ep, self.cst,  is_var_computed=True)
        STAT_u = mmd_val / torch.sqrt(mmd_var+10**(-8)) 
        return -STAT_u
    
    # compute the witness scores
    def compute_scores(self, X_ev, Y_ev, Z_input, batch_size=8192, max_loops=4):
        # adjust batch_size according to your memory capacity
        phi_Z = torch.zeros(Z_input.shape[0]).to(X_ev.device)
        X_ev_splited = torch.split(X_ev, batch_size)
        Y_ev_splited = torch.split(Y_ev, batch_size)
        Z_input_splited = torch.split(Z_input, batch_size)
        self.update_tool_params()
        for i_Z, Z_input_batch in enumerate(Z_input_splited):
            cum_sum = 0
            cum_count = 0
            for i_X in range(min(len(X_ev_splited),max_loops)):
                self.update_dist_matrix(X_ev_splited[i_X], Y_ev_splited[i_X], Z_input_batch, xz=True, yz=True)
                self.update_gram_matrix(X_ev_splited[i_X], Y_ev_splited[i_X], Z_input_batch, xz=True, yz=True)
                cum_sum += torch.sum(self.Kyz,0) - torch.sum(self.Kxz,0)
                cum_count += X_ev_splited[i_X].shape[0]
            phi_Z[i_Z*batch_size: i_Z*batch_size+Z_input_batch.shape[0]] = cum_sum/cum_count
        del X_ev_splited, Y_ev_splited, Z_input_splited;# gc.collect(); torch.cuda.empty_cache()
        return phi_Z
    
    # compute the threshold gamma
    def compute_gamma(self, X_ev, Y_ev, pi, batch_size=10000, MAX_LOOPS=100, verbose=False):
        # If X_ev and Y_ev are large, we use Monte Carlo 
        self.update_tool_params()
        n_ev = X_ev.shape[0]
        assert n_ev == Y_ev.shape[0]
        if n_ev < batch_size:
            self.update_dist_matrix(X_ev, Y_ev, None, xy=True, xz=False, yz=False, xx=True, yy=True)
            self.update_gram_matrix(X_ev, Y_ev, None, xy=True, xz=False, yz=False, xx=True, yy=True)
            EKxx = (torch.sum(self.Kxx) - torch.sum(torch.diag(self.Kxx)))/ (n_ev * (n_ev - 1))
            EKyy = (torch.sum(self.Kyy) - torch.sum(torch.diag(self.Kyy)))/ (n_ev * (n_ev - 1))
            EKxy = torch.sum(self.Kxy)/ (n_ev * n_ev)
            gamma = EKxx*(pi/2-1) + EKxy*(1-pi) + EKyy*(pi/2)
        else:
            gamma_records = torch.zeros(MAX_LOOPS)
            for i_monte in range(MAX_LOOPS):
                idx = np.random.choice(n_ev, batch_size, replace=False)
                idy = np.random.choice(n_ev, batch_size, replace=False)
                X_ev_batch = X_ev[idx]
                Y_ev_batch = Y_ev[idy]
                self.update_dist_matrix(X_ev_batch, Y_ev_batch, None, xy=True, xz=False, yz=False, xx=True, yy=True)
                self.update_gram_matrix(X_ev_batch, Y_ev_batch, None, xy=True, xz=False, yz=False, xx=True, yy=True)
                EKxx = (torch.sum(self.Kxx) - torch.sum(torch.diag(self.Kxx)))/ (batch_size * (batch_size - 1))
                EKyy = (torch.sum(self.Kyy) - torch.sum(torch.diag(self.Kyy)))/ (batch_size * (batch_size - 1))
                EKxy = torch.sum(self.Kxy) / (batch_size * batch_size)
                gamma_records[i_monte] = EKxx*(pi/2-1) + EKxy*(1-pi) + EKyy*(pi/2)
            if verbose:
                print('std/mean of Monte Carlo:', torch.abs(torch.std(gamma_records)/torch.mean(gamma_records)/np.sqrt(MAX_LOOPS)))
            gamma = torch.mean(gamma_records)
        return gamma
