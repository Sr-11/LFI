import numpy as np
import torch
import sys
from sklearn.utils import check_random_state
from utils import MatConvert, MMDu, TST_MMD_u, mmd2_permutations, MMD_General
from matplotlib import pyplot as plt
from tqdm import trange
import pickle
def sample_blobs_Q(N1, sigma_mx_2, rows=3, cols=3, rs=None):
    """Generate Blob-D for testing type-II error (or test power)."""
    """ Return: X,Y   """
    """ X ~ N(0, 0.03) + randint """
    """ Y ~ N(0, sigma_mx_2(9*2*2)) + {0,1,2}^2 """
    rs = check_random_state(rs)
    mu = np.zeros(2)
    sigma = np.eye(2) * 0.03
    X = rs.multivariate_normal(mu, sigma, size=N1)
    Y = rs.multivariate_normal(mu, np.eye(2), size=N1)
    # assign to blobs
    X[:, 0] += rs.randint(rows, size=N1)
    X[:, 1] += rs.randint(cols, size=N1)
    Y_row = rs.randint(rows, size=N1)
    Y_col = rs.randint(cols, size=N1)
    locs = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    for i in range(9):
        corr_sigma = sigma_mx_2[i]
        L = np.linalg.cholesky(corr_sigma) # corr_sigma=L*L^dagger, L is lower-triangular.
        ind = np.expand_dims((Y_row == locs[i][0]) & (Y_col == locs[i][1]), 1)
        ind2 = np.concatenate((ind, ind), 1)
        Y = np.where(ind2, np.matmul(Y,L) + locs[i], Y)
    return X, Y

class ModelLatentF(torch.nn.Module):
    """Latent space for both domains."""
    """ Dense Net with w=50, d=4, ~relu, in=2, out=50 """
    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant

def LFI_plot(n_list, title="LFI_with_Blob" ,path='./', with_error_bar=True):
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

def mmd(X, Y, model_u, n, m, sigma, sigma0_u, device, dtype, ep):
    S = np.concatenate((X, Y), axis=0)
    S = MatConvert(S, device, dtype)
    Fea = model_u(S)
    len_s = X.shape[0]
    return MMDu(Fea, len_s, S, sigma, sigma0_u, ep)

def mmdG(X, Y, model_u, n, m, sigma, sigma0_u, device, dtype, ep):
    S = np.concatenate((X, Y), axis=0)
    S = MatConvert(S, device, dtype)
    Fea = model_u(S)
    n = X.shape[0]
    m = Y.shape[0]
    return MMD_General(Fea, n, m, S, sigma, sigma0_u, ep)

def train(n_list, m_list, N_per=100, title='Default', alpha=0.05, learning_rate=5e-4, K=15, N=200, N_epoch=1000, print_every=100):  
  # Setup seeds
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
    with open('./PARAMETERS_'+title, 'wb') as pickle_file:
        pickle.dump(parameters, pickle_file)
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
        ep_OPT = np.zeros([K])
        s_OPT = np.zeros([K])
        s0_OPT = np.zeros([K])
        for kk in range(K):
            # Generate Blob-D
            X, Y = sample_blobs_Q(n, sigma_mx_2)
            S = np.concatenate((X, Y), axis=0)
            S = MatConvert(S, device, dtype)
            model_u = ModelLatentF(x_in, H, x_out).cuda()
            epsilonOPT = MatConvert(np.random.rand(1) * (10 ** (-10)), device, dtype)
            epsilonOPT.requires_grad = True
            sigmaOPT = MatConvert(np.sqrt(np.random.rand(1) * 0.3), device, dtype)
            sigmaOPT.requires_grad = True
            sigma0OPT = MatConvert(np.sqrt(np.random.rand(1) * 0.002), device, dtype)
            sigma0OPT.requires_grad = True
            # Setup optimizer for training deep kernel
            optimizer_u = torch.optim.Adam(list(model_u.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT], lr=learning_rate)
            # Train deep kernel to maximize test power
            for t in range(N_epoch):
                # Compute epsilon, sigma and sigma_0
                ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2
                # Compute output of the deep network
                modelu_output = model_u(S)
                # Compute J (STAT_u)
                TEMP = MMDu(modelu_output, n, S, sigma, sigma0_u, ep)
                mmd_value_temp = -1 * TEMP[0]
                mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                J_star_u[kk, t] = STAT_u.item()
                # Initialize optimizer and Compute gradient
                optimizer_u.zero_grad()
                STAT_u.backward(retain_graph=True)
                # Update weights using gradient descent
                optimizer_u.step()
                # Print MMD, std of MMD and J
                if t % print_every == 0:
                    print('Epoch:', t)
                    print("mmd_value: ", -1 * mmd_value_temp.item()) 
                          #"mmd_std: ", mmd_std_temp.item(), 
                    print("Statistic J: ", -1 * STAT_u.item())
            h_u, threshold_u, mmd_value_u = TST_MMD_u(model_u(S), N_per, n, S, sigma, sigma0_u, alpha, device, dtype, ep)
            ep_OPT[kk] = ep.item()
            s_OPT[kk] = sigma.item()
            s0_OPT[kk] = sigma0_u.item()
 
            #testing how model behaves on untrained data
            print('TEST OUR MODEL ON NEW SET OF DATA:')            
            X1, Y1 = sample_blobs_Q(n, sigma_mx_2)
            with torch.torch.no_grad():
                S = np.concatenate((X1, Y1), axis=0)
                S = MatConvert(S, device, dtype)
                modelu_output = model_u(S)
                TEMP = MMDu(modelu_output, n, S, sigma, sigma0_u, ep)
                mmd_value_temp = -1 * TEMP[0]
                mmd_std_temp = torch.sqrt(TEMP[1]+10**(-8))
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                J_star_u[kk, t] = STAT_u.item()
                if True:
                    print("TEST mmd_value: ", -1 * mmd_value_temp.item()) 
                          #"TEST mmd_std: ", mmd_std_temp.item(), 
                    print("TEST Statistic J: ", -1 * STAT_u.item())
            # Compute test power of deep kernel based MMD 
            
            #print(ep, epsilonOPT)
            print('epsilon:', ep)
            print('sigma:  ', sigma)
            print('sigma0: ', sigma0_u) 
            H_u = np.zeros(N) # 1 stands for correct, 0 stands for wrong
            print("Under this trained kernel, we run N = %d times: "%N)
            for k in range(N):
                #X, Y = sample_blobs_Q(n, sigma_mx_2)
                Z, _ = sample_blobs_Q(m, sigma_mx_2)
                # Run MMD on generated data
                mmd_XZ = mmdG(X, Z, model_u, n, m, sigma, sigma0_u, device, dtype, ep)
                mmd_YZ = mmdG(Y, Z, model_u, n, m, sigma, sigma0_u, device, dtype, ep)
                # Gather results
                H_u[k] = mmd_XZ<mmd_YZ    
            # Print probability of success
            print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~X): ", H_u.sum()/N_f)
            Results[0, kk] = H_u.sum() / N_f
            #raise KeyboardInterrupt
            for k in range(N):
                X, Y = sample_blobs_Q(n, sigma_mx_2)
                _, Z = sample_blobs_Q(m, sigma_mx_2)
                # Run MMD on generated data
                mmd_XZ = mmdG(X, Z, model_u, n, m, sigma, sigma0_u, device, dtype, ep)
                mmd_YZ = mmdG(Y, Z, model_u, n, m, sigma, sigma0_u, device, dtype, ep)
                # Gather results
                H_u[k] = mmd_XZ[0]>mmd_YZ[0]
              # Print probability of success
            print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~Y): ", H_u.sum()/N_f)
            Results[1, kk] = H_u.sum() / N_f
        np.save('./LFI_'+str(n),Results) 
    ####Plotting    
    LFI_plot(n_list, title=title)

if __name__ == "__main__":
    n_list = 10*np.array(range(8,11)) # number of samples in per mode
    m_list = 5*np.array(range(8,11))
    title=sys.argv[1]
    train(n_list, m_list, title=title)