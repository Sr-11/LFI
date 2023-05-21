
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
    def __init__(self, device, median_heuristic_flag=True, XY_heu=None, V=None, **kwargs):
        super(Model, self).__init__()
        # initialize parameters
        self.device = device
        self.model = DN().to(device)
        self.another_model = another_DN().to(device)
        self.epsilonOPT = MatConvert(np.zeros(1), device, dtype); self.epsilonOPT.requires_grad = True
        if median_heuristic_flag:
            with torch.no_grad():
                self.sigmaOPT = median(self.another_model(XY_heu), self.another_model(XY_heu))
                self.sigma0OPT = median(self.model(XY_heu), self.model(XY_heu))
                print('median_heuristic: sigma_0, sigma =', self.sigmaOPT, self.sigma0OPT)  
        else:
            self.sigmaOPT = MatConvert(np.sqrt([1.0]), device, dtype)
            self.sigma0OPT = MatConvert(np.sqrt([1.0]), device, dtype)
        self.sigmaOPT.requires_grad = True;  self.sigma0OPT.requires_grad = True
        self.cst = MatConvert(np.ones((1,)), device, dtype); self.cst.requires_grad = False
        self.V = V; self.V.requires_grad = True
        self.params = list(self.model.parameters())+list(self.another_model.parameters())+[self.epsilonOPT]+[self.sigmaOPT]+[self.sigma0OPT]+[self.cst]
        # for storing intermediate results
        self.ep = 0; self.sigma = 0; self.sigma0 = 0
        self.Dxx=torch.ones(1); self.Dxx_org=torch.ones(1); self.Dyy=torch.ones(1); self.Dyy_org=torch.ones(1); self.Dxy=torch.ones(1); self.Dxy_org=torch.ones(1); self.Dxz=torch.ones(1); self.Dxz_org=torch.ones(1); self.Dyz=torch.ones(1); self.Dyz_org=torch.ones(1)
        self.Dxv=torch.ones(1); self.Dxv_org=torch.ones(1); self.Dyv=torch.ones(1); self.Dyv_org=torch.ones(1); self.Dzv=torch.ones(1); self.Dzv_org=torch.ones(1)
        self.Kxx=torch.ones(1); self.Kyy=torch.ones(1); self.Kxy=torch.ones(1); self.Kxz=torch.ones(1); self.Kyz=torch.ones(1); self.Kxv=torch.ones(1); self.Kyv=torch.ones(1); self.Kzv=torch.ones(1)
    # compute the gram matrix. 
    def update_tool_params(self):
        self.ep = torch.exp(self.epsilonOPT)/(1+torch.exp(self.epsilonOPT))
        self.sigma = self.sigmaOPT ** 2; self.sigma0 = self.sigma0OPT ** 2
    def update_dist_matrix(self, X=None, Y=None, Z=None, xy=False, xz=False, yz=False, xx=False, yy=False, xv=False, yv=False, zv=False):
        if xy == True:
            self.Dxy = Pdist2_(self.Dxy, self.model(X), self.model(Y))
            self.Dxy_org = Pdist2_(self.Dxy_org, self.another_model(X), self.another_model(Y))
        if xz == True:
            self.Dxz = Pdist2_(self.Dxz, self.model(X), self.model(Z))
            self.Dxz_org = Pdist2_(self.Dxz_org, self.another_model(X), self.another_model(Z))
        if yz == True:
            self.Dyz = Pdist2_(self.Dyz, self.model(Y), self.model(Z))
            self.Dyz_org = Pdist2_(self.Dyz_org, self.another_model(Y), self.another_model(Z))
        if xx == True:
            self.Dxx = Pdist2_(self.Dxx, self.model(X), self.model(X))
            self.Dxx_org = Pdist2_(self.Dxx_org, self.another_model(X), self.another_model(X))
        if yy == True:
            self.Dyy = Pdist2_(self.Dyy, self.model(Y), self.model(Y))
            self.Dyy_org = Pdist2_(self.Dyy_org, self.another_model(Y), self.another_model(Y))
        # Don't use in-place
        if xv == True:
            self.Dxv = Pdist2(self.model(X), self.model(self.V))
            self.Dxv_org = Pdist2(self.another_model(X), self.another_model(self.V))
        if yv == True:
            self.Dyv = Pdist2(self.model(Y), self.model(self.V))
            self.Dyv_org = Pdist2(self.another_model(Y), self.another_model(self.V))
        if zv == True:
            self.Dzv = Pdist2(self.model(Z), self.model(self.V))
            self.Dzv_org = Pdist2(self.another_model(Z), self.another_model(self.V))
    def update_gram_matrix(self, X=None, Y=None, Z=None, xy=False, xz=False, yz=False, xx=False, yy=False, xv=False, yv=False, zv=False):
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
        # Don't use in-place
        if xv == True:
            self.Kxv = self.cst*((1-self.ep)*torch.exp(-self.Dxv/self.sigma0)+self.ep)*torch.exp(-self.Dxv_org/self.sigma)
        if yv == True:
            self.Kyv = self.cst*((1-self.ep)*torch.exp(-self.Dyv/self.sigma0)+self.ep)*torch.exp(-self.Dyv_org/self.sigma)
        if zv == True:
            self.Kzv = self.cst*((1-self.ep)*torch.exp(-self.Dzv/self.sigma0)+self.ep)*torch.exp(-self.Dzv_org/self.sigma)
         
    # compute loss
    def clamp(self):
        with torch.no_grad():
            self.cst.clamp_(min=0.5, max=2.0)
            self.epsilonOPT.clamp_(min=-10.0, max=10.0)
            self.sigmaOPT.clamp_(min=0.0, max=30.0)
            self.sigma0OPT.clamp_(min=0.0, max=30.0)
    def compute_UME(self, XY_tr):
        # compute feature matrix
        self.update_tool_params()
        nx = XY_tr.shape[0]//2
        self.update_dist_matrix(X=XY_tr[:nx], Y=XY_tr[nx:], xv=True, yv=True)
        self.update_gram_matrix(X=XY_tr[:nx], Y=XY_tr[nx:], xv=True, yv=True)
        J = self.V.shape[0]
        feature_matrix = 1/np.sqrt(J) * (self.Kxv - self.Kyv) # See definition in the UME paper
        # compute the mean 
        t1 = torch.sum(torch.mean(feature_matrix, axis=0)**2) * (nx/float(nx-1))
        t2 = torch.mean(torch.sum(feature_matrix**2, axis=1)) / float(nx-1)
        UME_mean = t1 - t2
        # compute the variance
        mu = torch.mean(feature_matrix, axis=0, keepdim=True).t() # 1*J vector
        UME_variance = 4.0*torch.mean(torch.matmul(feature_matrix, mu)**2) - 4.0*torch.sum(mu**2)**2
        return UME_mean, UME_variance
    def compute_loss(self, XY_tr):
        # self.clamp()
        UME_mean, UME_var = self.compute_UME(XY_tr)
        UME_std = torch.sqrt(UME_var+10**(-6))
        ratio = torch.div(UME_mean,UME_std)
        return -ratio
    
    # compute witness scores
    def compute_scores(self, X_ev, Y_ev, Z_input, batch_size=4096, max_loops=1000):
        self.update_tool_params()
        Z_input_splited = torch.split(Z_input, batch_size)
        X_ev_splited = torch.split(X_ev, batch_size)
        Y_ev_splited = torch.split(Y_ev, batch_size)
        J = self.V.shape[0]
        result = torch.zeros(Z_input.shape[0]).to(X_ev.device)
        for i_Z, Z_input_batch in enumerate(Z_input_splited):
            feature_cum_sum = 0
            cum_cnt = 0
            for i_X in range(min(len(X_ev_splited), max_loops)):
                X_batch = X_ev_splited[i_X]
                Y_batch = Y_ev_splited[i_X]
                self.update_dist_matrix(X=X_batch, Y=Y_batch, Z=Z_input_batch, xv=True, yv=True, zv=True)
                self.update_gram_matrix(X=X_batch, Y=Y_batch, Z=Z_input_batch, xv=True, yv=True, zv=True)
                feature_matrix = 1/np.sqrt(J) * (self.Kxv - self.Kyv) # See definition in the UME paper
                feature_cum_sum += torch.sum(feature_matrix, dim=0)
                cum_cnt += feature_matrix.shape[0]
            result[i_Z*batch_size: i_Z*batch_size+Z_input_batch.shape[0]] = -2/np.sqrt(J) * torch.matmul(feature_cum_sum/cum_cnt, self.Kzv.t())
            gc.collect(); torch.cuda.empty_cache()
        return result
