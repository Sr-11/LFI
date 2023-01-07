import numpy as np
import torch
import sys
from sklearn.utils import check_random_state
from utils import *
from matplotlib import pyplot as plt
import pickle
from Data_gen import *

class ModelLatentF(torch.nn.Module):
    """Latent space for both domains."""
    """ Dense Net with w=50, d=4, ~relu, in=2, out=50 """
    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant

def crit(mmd_val, mmd_var, liuetal=True, Sharpe=False):
    """compute the criterion."""
    ######IMPORTANT: if we want to maximize, need to multiply by -1######
    if liuetal:
        mmd_std_temp = torch.sqrt(mmd_var+10**(-8)) #this is std
        return torch.div(mmd_val, mmd_std_temp)
    elif Sharpe:
        return mmd_val - 2.0 * mmd_var

def mmdG(X, Y, model_u, n, sigma, sigma0_u, device, dtype, ep):
    S = np.concatenate((X, Y), axis=0)
    S = MatConvert(S, device, dtype)
    Fea = model_u(S)
    n = X.shape[0]
    return MMD_General(Fea, n, S, sigma, sigma0_u, ep)

def train_d(n, m_list, title='Default', learning_rate=5e-4, K=10, N=1000, N_epoch=500, print_every=100, batch_size=32, test_on_new_sample=True, SGD=True, gen_fun=blob):  
    torch.backends.cudnn.deterministic = True
    dtype = torch.float
    device = torch.device("cuda:0")
    x_in = 2 # number of neurons in the input layer, i.e., dimension of data
    H = 50 # number of neurons in the hidden layer
    x_out = 50 # number of neurons in the output layer
    N_f = float(N) # number of test sets (float)
    if not SGD:
        batch_size=n
        batches=1
    else:
        batches=1+n//batch_size
        n=batches*batch_size #round up
    parameters={'n':n,
                'm_list':m_list,
                'N_epoch':N_epoch,
                'learning_rate':learning_rate,
                'batch_size':batch_size,
                'batches':batches,
                'test_on_new_sample':test_on_new_sample,
                'SGD':SGD,
                'gen_fun':gen_fun(-1),
                'K':K,
                'N':N,}
    with open('./data/PARAMETERS_'+title, 'wb') as pickle_file:
        pickle.dump(parameters, pickle_file)
    
    print("##### Starting N_epoch=%d epochs per data trial #####"%(N_epoch))
    print("##### K=%d independent kernels, N=%d tests per trial for inference of Z per m. #####"%(K,N))
    if test_on_new_sample:
        print("We test on new samples x, y not during training")
    else:
        print("We reuse samples x, y during training")
    Results = np.zeros([2, K, len(m_list)]) ###Result[{0, 1}, K, m] where K is an index
    J_star_u = np.zeros([K, N_epoch])
    ep_OPT = np.zeros([K])
    s_OPT = np.zeros([K])
    s0_OPT = np.zeros([K])
    for kk in range(K):
            X, Y = gen_fun(n)
            total_S=[(X[i*batch_size:i*batch_size+batch_size], Y[i*batch_size:i*batch_size+batch_size]) for i in range(batches)]
            total_S=[MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in total_S]
            model_u = ModelLatentF(x_in, H, x_out).cuda()
            epsilonOPT = MatConvert(np.random.rand(1) * (10 ** (-10)), device, dtype)
            epsilonOPT.requires_grad = True
            sigmaOPT = MatConvert(np.sqrt(np.random.rand(1) * 0.3), device, dtype)
            sigmaOPT.requires_grad = True
            sigma0OPT = MatConvert(np.sqrt(np.random.rand(1) * 0.002), device, dtype)
            sigma0OPT.requires_grad = True
            eps=MatConvert(np.zeros((1,)), device, dtype)
            eps.requires_grad = True
            optimizer_u = torch.optim.Adam(list(model_u.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT]+[eps], lr=learning_rate)

            for t in range(N_epoch):
                # Compute epsilon, sigma and sigma_0
                for ind in range(batches):
                    ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
                    sigma = sigmaOPT ** 2
                    sigma0_u = sigma0OPT ** 2
                    S=total_S[ind]
                    modelu_output = model_u(S)
                    TEMP = MMDu(modelu_output, batch_size, S, sigma, sigma0_u, ep)
                    mmd_val=-1 * TEMP[0]
                    mmd_var=TEMP[1]
                    STAT_u=crit(mmd_val, mmd_var)
                    J_star_u[kk, t] = STAT_u.item()
                    optimizer_u.zero_grad()
                    STAT_u.backward(retain_graph=True)
                    optimizer_u.step()
                # Print MMD, std of MMD and J
                if t % print_every == 0:
                    print('Epoch:', t)
                    print("mmd_value: ", mmd_val.item()) 
                          #"mmd_std: ", mmd_std_temp.item(), 
                    print("Statistic J: ", STAT_u.item())
            ep_OPT[kk] = ep.item()
            s_OPT[kk] = sigma.item()
            s0_OPT[kk] = sigma0_u.item()
            
            #testing how model behaves on untrained data
            print('CRITERION ON NEW SET OF DATA:')            
            X1, Y1 = gen_fun(n)
            with torch.torch.no_grad():
                S1 = np.concatenate((X1, Y1), axis=0)
                S1 = MatConvert(S1, device, dtype)
                modelu_output = model_u(S1)
                TEMP = MMDu(modelu_output, n, S1, sigma, sigma0_u, ep)
                mmd_value_temp, mmd_var_temp = TEMP[0], TEMP[1]
                STAT_u = crit(mmd_value_temp, mmd_var_temp)
                if True:
                    print("TEST mmd_value: ", mmd_value_temp.item()) 
                    print("TEST Statistic J: ", STAT_u.item())
            H_u = np.zeros(N) 
            print("Under this trained kernel, we run N = %d times LFI: "%N)
            if test_on_new_sample:
                X, Y = gen_fun(n)
            for i in range(len(m_list)):
                m = m_list[i]
                for k in range(N):       
                    Z, _ = gen_fun(m)
                    # Run MMD on generated data
                    mmd_XZ = mmdG(X, Z, model_u, n, sigma, sigma0_u, device, dtype, ep)[0]
                    mmd_YZ = mmdG(Y, Z, model_u, n, sigma, sigma0_u, device, dtype, ep)[0]
                    H_u[k] = mmd_XZ<mmd_YZ    
                print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~X): ", H_u.sum()/N_f)
                Results[0, kk, i] = H_u.sum() / N_f

                for k in range(N):
                    _, Z = gen_fun(m)
                    mmd_XZ = mmdG(X, Z, model_u, n, sigma, sigma0_u, device, dtype, ep)[0]
                    mmd_YZ = mmdG(Y, Z, model_u, n, sigma, sigma0_u, device, dtype, ep)[0]
                    H_u[k] = mmd_XZ>mmd_YZ
                print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~Y): ", H_u.sum()/N_f)
                Results[1, kk, i] = H_u.sum() / N_f
    np.save('./data/LFI_tst_'+title+str(n),Results) 
    ####Plotting    
    #LFI_plot(n_list, title=title)

def train_O(n_list, m_list):
    #Trains optimized Gaussian Kernel Length
    #implemented in DK-TST
    pass

if __name__ == "__main__":
    n=100
    m_list = 10*np.array(range(4,5))
    try:
        title=sys.argv[1]
    except:
        print("Warning: No title given, using default")
        print('Please use specified titles for saving data')
        title='untitled_run'
    train_d(n, m_list, title=title, learning_rate=5e-4, K=10, N=1000, N_epoch=500, print_every=100, batch_size=32, test_on_new_sample=True, SGD=True, gen_fun=blob)