import numpy as np
import torch
import sys
from sklearn.utils import check_random_state
from utils import *
from matplotlib import pyplot as plt
import pickle
from Data_gen import *
import torch.nn as nn

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

# Note: in DK_TST, they first interpolate CIFAR10 from 32x32 to 64x64
# Here input should be (N,3x64x64), dim=3*64*64 
class ConvNet_CIFAR10(nn.Module):
    """
    input: (N,3x64x64)
    output: (N,300)
    """
    def __init__(self):
        super(ConvNet_CIFAR10, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block =([nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=True),  
                     nn.Dropout2d(0)]) #0.25
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        self.model = nn.Sequential(
            nn.Unflatten(1,(3,64,64)),
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        # The height and width of downsampled image
        ds_size = 64 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 300))
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)
        return feature

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

def train_d(n, m_list, title='Default', learning_rate=5e-4, K=10, N=1000, N_epoch=50, 
            print_every=100, batch_size=32, test_on_new_sample=True, SGD=True, gen_fun=blob, seed=42):  
    #set random seed for torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    dtype = torch.float
    device = torch.device("cuda:0")
    x_in = 2 # number of neurons in the input layer, i.e., dimension of data, blob:2
    # for blob, use x_in=2, H=50, x_out=50
    # for cifar10, use x_in=3072
    #x_in = 3072 # number of neurons in the input layer, i.e., dimension of data, CIFAR10:32*32*3=3072
    H = 50 # number of neurons in the hidden layers
    x_out = 50 # number of neurons in the output layer (feature space)
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
                'seed' : seed,
                'N':N,}
    with open('./data/PARAMETERS_'+title, 'wb') as pickle_file:
        pickle.dump(parameters, pickle_file)
    
    print("\n#############################################")
    print("##### Starting N_epoch=%d epochs per data trial #####"%(N_epoch))
    print("##### K=%d independent kernels, N=%d tests per trial for inference of Z per m. #####"%(K,N))
    print("#############################################\n")

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
        total_S=[(X[i*batch_size:i*batch_size+batch_size], 
                    Y[i*batch_size:i*batch_size+batch_size]) 
                    for i in range(batches)]
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
                print('------------------------------------')
                print('Epoch:', t)
                print("mmd_value: ", mmd_val.item()) 
                        #"mmd_std: ", mmd_std_temp.item(), 
                print("Statistic J: ", STAT_u.item())
                print('------------------------------------')
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
        H_v = np.zeros(N)
        print("Under this trained kernel, we run N = %d times LFI: "%N)
        if test_on_new_sample:
            X, Y = gen_fun(n)
        for i in range(len(m_list)):
            print("start testing m = %d"%m_list[i])
            m = m_list[i]
            for k in range(N):       
                Z1, Z2 = gen_fun(m)
                mmd_XZ = mmdG(X, Z1, model_u, n, sigma, sigma0_u, device, dtype, ep)[0]
                mmd_YZ = mmdG(Y, Z1, model_u, n, sigma, sigma0_u, device, dtype, ep)[0]
                H_u[k] = mmd_XZ<mmd_YZ    
                mmd_XZ = mmdG(X, Z2, model_u, n, sigma, sigma0_u, device, dtype, ep)[0]
                mmd_YZ = mmdG(Y, Z2, model_u, n, sigma, sigma0_u, device, dtype, ep)[0]
                H_v[k] = mmd_XZ>mmd_YZ
            print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~X): ", H_u.sum()/N_f)
            print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~Y): ", H_v.sum()/N_f)
            Results[0, kk, i] = H_u.sum() / N_f
            Results[1, kk, i] = H_v.sum() / N_f

    np.save('./data/LFI_tst_'+title+str(n),Results) 
    ####Plotting    
    #LFI_plot(n_list, title=title)

def train_O(n_list, m_list):
    #Trains optimized Gaussian Kernel Length
    #implemented in DK-TST
    pass

if __name__ == "__main__":
    n=500
    m_list = 10*np.array(range(4,5))
    random_seed=42
    try:
        title=sys.argv[1]
    except:
        print("Warning: No title given, using default")
        print('Please use specified titles for saving data')
        title='untitled_run'

    diffusion_data=True
    if diffusion_data:
        dataset_P, dataset_Q = load_diffusion_cifar() #Helper Function in Data_gen.py
        def diffusion_cifar10(n):
            if n <0 :
                return 'DIFFUSION'            
            np.random.shuffle(dataset_P)
            Xs = dataset_P[:n]
            np.random.shuffle(dataset_Q)
            Ys = dataset_Q[:n]
            return Xs, Ys
    train_d(n, m_list, title=title, learning_rate=5e-4, K=100, N=1000, 
            N_epoch=1, print_every=100, batch_size=32, test_on_new_sample=False, 
            SGD=True, gen_fun=blob, seed=random_seed)
    # n: size of X, Y
    # m: size of Z
    # K: number of experiments, each with different X, Y
    # N: number of runs for LFI, each with different Z

    # title: used for saving data
    # learning_rate: learning rate for training
    # N_epoch: number of epochs for training
    # print_every: print every print_every epochs during training
    # batch_size: batch size for training
    # test_on_new_sample: if Flase, use the same X, Y for test (overfit)
    # SGD: if True, use batch_size
    # gen_fun: from Data_gen.py