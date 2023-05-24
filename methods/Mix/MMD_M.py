import numpy as np
import torch
import sys, os, gc
from utils import *
from matplotlib import pyplot as plt
from GPUtil import showUtilization as gpu_usage
from tqdm import tqdm, trange
import config
import json
import inspect
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import lfi
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
torch.backends.cudnn.deterministic = True
dtype = torch.float
device = torch.device("cuda:0")
torch.manual_seed(42)
np.random.seed(42)

cur_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))
ckpts_dir = os.path.join(cur_dir, 'checkpoints')

# define network
H = 300
out = 100
x_in = 28
L = 1
class DN(torch.nn.Module):
    def __init__(self):
        super(DN, self).__init__()
        self.restored = False
        self.model = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
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
    def __init__(self, H=300, out=100):
        super(another_DN, self).__init__()
        self.restored = False
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, 28, bias=True),
        )
    def forward(self, input):
        output = self.model(input) + input
        return output

# define loss function
def crit(mmd_val, mmd_var, liuetal=True, Sharpe=False):
    if liuetal:
        mmd_std_temp = torch.sqrt(mmd_var+10**(-8)) #this is std
        return torch.div(mmd_val, mmd_std_temp)

# save checkpoint
def save_model(n,model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst,epoch=0, r=0):
    path = os.path.join(ckpts_dir, 'n_tr=%d#%d'%(n,r))
    try: os.makedirs(path) 
    except: pass
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
    torch.save(another_model.state_dict(), os.path.join(path, 'another_model.pt'))
    torch.save(epsilonOPT, os.path.join(path, 'epsilonOPT.pt'))
    torch.save(sigmaOPT, os.path.join(path, 'sigmaOPT.pt'))
    torch.save(sigma0OPT, os.path.join(path, 'sigma0OPT.pt'))
    torch.save(eps, os.path.join(path, 'eps.pt'))
    torch.save(cst, os.path.join(path, 'cst.pt'))

# train
def train(n, learning_rate=5e-4, 
            K=10, N=1000, N_epoch=50, 
            print_every=1, batch_size=32, 
            SGD=True, 
            dataset_P=None, dataset_Q=None,
            momentum = 0.9, weight_decay=0.0,
            r=None):  
    n_backup = n
    if n>batch_size*2:
        #cuda.select_device(0)
        batches = (n-1)//batch_size + 1 # last batch could be empty
        n = batches*batch_size  
        print("------------------------------------------------")
        X = dataset_P[0:n]
        Y = dataset_Q[0:n]
        print(X.shape, Y.shape)
        total_S = [(X[i*batch_size:(i+1)*batch_size], 
                    Y[i*batch_size:(i+1)*batch_size]) 
                    for i in range(batches)]
    else:
        batches = 1
        total_S = [(dataset_P[:n], dataset_Q[:n])] 
    total_S = [MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in total_S]
    # prepare NN
    model = DN().to(device)
    another_model = another_DN().to(device)
    # prepare other parameters
    epsilonOPT = MatConvert(-np.random.rand(1), device, dtype)
    # with torch.no_grad():
    #     sigma0OPT = lfi.utils.median(model(total_S[0]))
    #     sigmaOPT = lfi.utils.median(another_model(total_S[0]))
    # epsilonOPT.requires_grad = True
    # sigmaOPT.requires_grad = True
    
    sigmaOPT = MatConvert(np.sqrt(np.random.rand(1)), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.sqrt(np.random.rand(1)), device, dtype)
    sigma0OPT.requires_grad = True
    sigma0OPT.requires_grad = True
    
    eps=MatConvert(np.zeros((1,)), device, dtype)
    eps.requires_grad = True
    cst=MatConvert(1*np.ones((1,)), device, dtype)
    cst.requires_grad = False
    # prepare optimizer
    params = list(model.parameters())+list(another_model.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT]+[eps]+[cst]
    if SGD:
        optimizer_u = torch.optim.SGD(params, lr=learning_rate,
                                    momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer_u = torch.optim.Adam(params, lr=learning_rate)
    # validation data
    S1_v = np.concatenate((dataset_P[n + np.random.choice(n, min(int(0.1*n),int(np.sqrt(10*n))), replace=False)], 
                            dataset_Q[n + np.random.choice(n, min(int(0.1*n),int(np.sqrt(10*n))), replace=False)]), axis=0)
    S1_v = MatConvert(S1_v, device, dtype)
    J_validations = np.ones([N_epoch])*np.inf
    mmd_val_validations = np.zeros([N_epoch])
    #############################
    #############################
    ckpt_dir = os.path.join(ckpts_dir, 'n_tr=%d#%d'%(n_backup,r))
    validation_plot_path = os.path.join(ckpt_dir, 'validation.png')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    for t in range(N_epoch):
        print('n=%d, r=%d,  epoch=%d'%(n_backup,r,t))
        order = np.random.permutation(batches)
        for ind in tqdm(order):
            optimizer_u.zero_grad()
            ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
            sigma = sigmaOPT ** 2
            sigma0_u = sigma0OPT ** 2
            S = total_S[ind]
            modelu_output = model(S) 
            another_output = another_model(S)
            TEMP = MMDu(modelu_output, batch_size, another_output, sigma, sigma0_u, ep, cst, L=L) # could raise error
            mmd_val = -1 * TEMP[0]
            mmd_var = TEMP[1]
            STAT_u = crit(mmd_val, mmd_var) 
            STAT_u.backward(retain_graph=False)
            optimizer_u.step()      
        # validation
        with torch.torch.no_grad():
            modelu_output = model(S1_v)
            another_output = another_model(S1_v)
            TEMP = MMDu(modelu_output, modelu_output.shape[0]//2, another_output, sigma, sigma0_u, ep, cst, L=L)
            mmd_value_temp, mmd_var_temp = -TEMP[0], TEMP[1]
            mmd_val_validations[t] = mmd_value_temp.item()
            J_validations[t] = crit(mmd_value_temp, mmd_var_temp).item()
            print('validation∂ =', J_validations[t])
            #print(sigmaOPT.item(), sigma0OPT.item(), epsilonOPT.item())
        if t%print_every==0:
            plt.plot(J_validations[:t])
            plt.savefig(validation_plot_path)
            plt.clf()
            save_model(n_backup,model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst,epoch=t, r=r)
                
        if early_stopping(J_validations, t) and J_validations[t]<-0.1:
            plt.plot(J_validations[:t])
            plt.savefig(validation_plot_path)
            plt.clf()
            save_model(n_backup,model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,eps,cst,epoch=0, r=r)
            return model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,J_validations[t]
    return model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,J_validations[t]


if __name__ == "__main__":
    current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))) 

    # load data, please use .npy ones (40000), .gz (10999999)(11e6) is too large.
    dataset = np.load(os.path.join(current_dir, '..', '..', 'datasets', 'HIGGS.npy'))
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)

    n_tr_list = config.train_param_configs['n_tr_list']
    repeat = config.train_param_configs['repeat']
    os.environ["CUDA_VISIBLE_DEVICES"]= config.train_param_configs['gpu']

    kwargs = dict([arg.split('=') for arg in sys.argv[1:]])
    if 'n_tr_list' in kwargs:
        n_tr_list = json.loads(kwargs['n_tr_list'])
    if 'repeat' in kwargs:
        repeat = json.loads(kwargs['repeat'])
    if 'gpu' in kwargs:
        os.environ["CUDA_VISIBLE_DEVICES"]= kwargs['gpu']

    for n in n_tr_list:
        for r in repeat:
            gc.collect()
            torch.cuda.empty_cache()
            while True:
                print('-------------------')
                print(n)
                model,another_model,epsilonOPT,sigmaOPT,sigma0OPT,cst,J = train(n, 
                                                                N_epoch = 101, # 只load就设成1
                                                                print_every = 1, 
                                                                batch_size = 1024, 
                                                                learning_rate =2e-3, 
                                                                SGD = True, 
                                                                dataset_P = dataset_P, dataset_Q = dataset_Q, #Mixture
                                                                momentum=0.99, weight_decay=0.000, r=r)
                if J<-0.1:
                    break
        