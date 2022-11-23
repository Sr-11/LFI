import numpy as np
import torch
import sys
from sklearn.utils import check_random_state
from utils import MatConvert, MMDu, TST_MMD_u, mmd2_permutations, MMD_General, MMD_LFI_std
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

def LFI_plot(n_list, title="LFI_with_Blob" ,path='./data/', with_error_bar=True):
    Results_list = []
    fig = plt.figure(figsize=(10, 8))
    ZX_success = np.zeros(len(n_list))
    ZY_success = np.zeros(len(n_list))
    ZX_err = np.zeros(len(n_list))
    ZY_err = np.zeros(len(n_list))
    for i,n in enumerate(n_list):
        Results_list.append(np.load(path+'LFI_%d.npy'%n))
        ZX_success[i] = np.mean(Results_list[-1][0,:])
        ZY_success[i] = np.mean(Results_list[-1][1,:])
        ZX_err[i] = np.std(Results_list[-1][0,:])
        ZY_err[i] = np.std(Results_list[-1][1,:])
    if with_error_bar==True:
        plt.errorbar(n_list, ZX_success, yerr=ZX_err, label='Z~X')
        plt.errorbar(n_list, ZY_success, yerr=ZY_err, label='Z~Y')
    else:
        plt.plot(n_list, ZX_success, label='Z~X')
        plt.plot(n_list, ZY_success, label='Z~Y')
    print('Success rates:')
    print(ZX_success)
    print(ZY_success)
    plt.xlabel("n samples", fontsize=20)
    plt.ylabel("P(success)", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(title+'.png')
    plt.close()
    return fig

def mmdG(X, Y, model_u, n, m, device, dtype):
    S = np.concatenate((X, Y), axis=0)
    S = MatConvert(S, device, dtype)
    Fea = model_u(S)
    n = X.shape[0]
    m = Y.shape[0]
    return MMD_General(Fea, n, m, S)

def train_d(n_list, m_list, N_per=100, title='Default', learning_rate=5e-4, K=15, N=1000, N_epoch=51, print_every=100, batch_size=50, test_on_new_sample=True, SGD=False, LfI=False):  
    torch.backends.cudnn.deterministic = True
    dtype = torch.float
    device = torch.device("cuda:0")
    x_in = 2 # number of neurons in the input layer, i.e., dimension of data
    H = 50 # number of neurons in the hidden layer
    x_out = 50 # number of neurons in the output layer
    N_f = float(N) # number of test sets (float)
    parameters={'n_list':n_list,
                'm_list':m_list,
                'N_epoch':N_epoch,
                'N_per':N_per,
                'K':K,
                'N':N,}
    #with open('./data/PARAMETERS_'+title, 'wb') as pickle_file:
    #    pickle.dump(parameters, pickle_file)
    # Generate variance and co-variance matrix of Q
    sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
    sigma_mx_2 = np.zeros([9,2,2])
    for i in range(9):
        sigma_mx_2[i] = sigma_mx_2_standard
        if i < 4:
            sigma_mx_2[i][0 ,1] = -0.02 - 0.002 * i
            sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
        if i==4:
            sigma_mx_2[i][0, 1] = 0.00
            sigma_mx_2[i][1, 0] = 0.00
        if i>4:
            sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i-5)
            sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i-5)

    for i in range(len(n_list)):
        n=n_list[i]
        m=m_list[i]
        print("##### Starting n=%d and m=%d #####"%(n, m))
        print("##### Starting N_epoch=%d epochs #####"%(N_epoch))
        print("##### K=%d big trials, N=%d tests per trial for inference of Z. #####"%(K,N))
        Results = np.zeros([2, K])
        J_star_u = np.zeros([K, N_epoch])
        for kk in range(K):
            if not SGD:
                batch_size=n
                batch_m=m
            else:
                batches=1+n//batch_size
                n=batches*batch_size #round up
                m=(m//batches)*batches
                batch_m=m//batches
            X, Y = sample_blobs_Q(n, sigma_mx_2)
            Z, _ = sample_blobs_Q(m, sigma_mx_2)
            total_S=[(X[i*batch_size:i*batch_size+batch_size], Y[i*batch_size:i*batch_size+batch_size]) for i in range(batches)]
            total_Z=[Z[i*batch_m:i*batch_m+batch_m] for i in range(batches)]
            model_u = ModelLatentF(x_in, H, x_out).cuda()
            optimizer_u = torch.optim.Adam(list(model_u.parameters()), lr=learning_rate)
            # Train deep kernel to maximize test power
            for t in range(N_epoch):
                # Compute epsilon, sigma and sigma_0
                for ind in range(batches):
                    x, y=total_S[ind] #minibatches
                    z=total_Z[ind]
                    if not LfI:
                        S=MatConvert(np.concatenate((x, y), axis=0), device, dtype)
                        modelu_output = model_u(S)
                        TEMP = MMDu(modelu_output, batch_size, S)
                        mmd_value_temp = -1 * TEMP[0]
                        mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8)) #this is std
                        STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                    else:
                        S=MatConvert(np.concatenate(([x, y, z]), axis=0), device, dtype)
                        Fea=model_u(S)
                        #mmd_value_temp = torch.square(mmdG(x, z, model_u, batch_size, batch_m, sigma, sigma0_u, device, dtype, ep)-mmdG(y, z, model_u, batch_size, batch_m, sigma, sigma0_u, device, dtype, ep))
                        #The above is correct form but very slow.
                        lfi_temp=MMD_LFI_std(Fea, S, batch_size, batch_m)
                        #TODO: compute std
                        mmd_squared_temp=torch.square(lfi_temp[0])
                        mmd_var_temp=torch.nn.ReLU(lfi_temp[1])
                        STAT_u = torch.sub(mmd_squared_temp, mmd_var_temp, alpha=1.0)
                    J_star_u[kk, t] = STAT_u.item()
                    optimizer_u.zero_grad()
                    STAT_u.backward(retain_graph=True)
                    optimizer_u.step()
                # Print MMD, std of MMD and J
                if t % print_every == 0:
                    print('Epoch:', t)
                    print("mmd_value: ", -1 * mmd_value_temp.item()) 
                          #"mmd_std: ", mmd_std_temp.item(), 
                    print("Statistic J: ", -1 * STAT_u.item())
            '''
            #testing how model behaves on untrained data
            print('TEST OUR MODEL ON NEW SET OF DATA:')            
            X1, Y1 = sample_blobs_Q(n, sigma_mx_2)
            with torch.torch.no_grad():
                S1 = np.concatenate((X1, Y1), axis=0)
                S1 = MatConvert(S1, device, dtype)
                modelu_output = model_u(S1)
                TEMP = MMDu(modelu_output, n, S1, sigma, sigma0_u, ep)
                mmd_value_temp = -1 * TEMP[0]
                mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                if True:
                    print("TEST mmd_value: ", -1 * mmd_value_temp.item()) 
                    print("TEST Statistic J: ", -1 * STAT_u.item())
            #print('epsilon:', ep)
            #print('sigma:  ', sigma)
            #print('sigma0: ', sigma0_u) 
            '''
            H_u = np.zeros(N) 
            print("Under this trained kernel, we run N = %d times LFI: "%N)
            for k in range(N):
                if test_on_new_sample:
                    X, Y = sample_blobs_Q(n, sigma_mx_2)
                Z, _ = sample_blobs_Q(m, sigma_mx_2)
                # Run MMD on generated data
                mmd_XZ = mmdG(X, Z, model_u, n, m, device, dtype)[0]
                mmd_YZ = mmdG(Y, Z, model_u, n, m, device, dtype)[0]
                H_u[k] = mmd_XZ<mmd_YZ    
            print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~X): ", H_u.sum()/N_f)
            Results[0, kk] = H_u.sum() / N_f
            for k in range(N):
                if test_on_new_sample:
                    X, Y = sample_blobs_Q(n, sigma_mx_2)
                _, Z = sample_blobs_Q(m, sigma_mx_2)
                mmd_XZ = mmdG(X, Z, model_u, n, m, device, dtype)[0]
                mmd_YZ = mmdG(Y, Z, model_u, n, m, device, dtype)[0]
                H_u[k] = mmd_XZ>mmd_YZ
            print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~Y): ", H_u.sum()/N_f)
            Results[1, kk] = H_u.sum() / N_f
        np.save('./data/LFI_'+str(n),Results) 
    ####Plotting    
    LFI_plot(n_list, title=title)

def train_O(n_list, m_list):
    #Trains optimized Gaussian Kernel Length
    #implemented in DK-TST
    pass

if __name__ == "__main__":
    n_list = 50*np.array(range(12,13)) # number of samples in per mode
    m_list = 10*np.array(range(4,5))
    try:
        title=sys.argv[1]
    except:
        title='untitled_run'
    train_d(n_list, m_list, title=title)