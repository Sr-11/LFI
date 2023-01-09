import numpy as np
import torch
import sys
from sklearn.utils import check_random_state
from utils import MatConvert, MMD_General, MMD_LFI_STAT, relu, MMD_STAT
from matplotlib import pyplot as plt
import pickle
from Data_gen import *
import torch.nn as nn
import time

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
# Here input should be (N,3x32x32), dim=3*32*32 
class ConvNet_CIFAR10(nn.Module):
    """
    input: (N,3x32x32)
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
            nn.Unflatten(1,(3,32,32)),
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4 # for 64*64 image, set ds_size = 64 // 2 ** 4
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

def LFI_plot(n_list, title="LFI_with_Blob" ,path='./data/', with_error_bar=True):
    Results_list = []
    fig = plt.figure(figsize=(10, 8))
    ZX_success = np.zeros(len(n_list))
    ZY_success = np.zeros(len(n_list))
    ZX_err = np.zeros(len(n_list))
    ZY_err = np.zeros(len(n_list))
    for i,n in enumerate(n_list):
        Results_list.append(np.load(path+'LFI_%d.npy'%n))
        print(Results_list)
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
    print('Z~X:',ZX_success)
    print('X~Y:',ZY_success)
    print('Variance of success rates:')
    print('Z~X:',ZX_err)
    print('X~Y:',ZY_err)
    plt.xlabel("n samples", fontsize=20)
    plt.ylabel("P(success)", fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig(title+'.png')
    plt.close()
    return fig

# NoteL MMD_General is in utils.py
def mmdG(X, Y, model_u, n, m, sigma, cst, device, dtype):
    S = np.concatenate((X, Y), axis=0)
    S = MatConvert(S, device, dtype)
    Fea = model_u(S)
    n = X.shape[0]
    m = Y.shape[0]
    return MMD_General(Fea, n, m, S, sigma, cst)

def train_d(n_list, m_list, title='Default', learning_rate=5e-4, 
            K=10, N=1000, N_epoch=50, print_every=100, batch_size=32, 
            test_on_new_sample=True, SGD=True, gen_fun=blob, seed=42):
    # deal with the n%batch_size problem
    print('-------------------')
    print('Use SGD:',SGD)
    print('n_list:',n_list)
    print('m_list:',m_list)
    n_list_rounded = []
    m_list_rounded = []
    for i in range(len(n_list)):
        n=n_list[i]
        m=m_list[i]
        batches=n//batch_size
        n=batches*batch_size #round up
        batch_m=m//batches
        m=(m//batches)*batches
        n_list_rounded.append(n)
        m_list_rounded.append(m)
    if SGD==True:
        print('n_list_rounded:',n_list_rounded)
        print('m_list_rounded:',m_list_rounded)
    print('-------------------')

    #set random seed for torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    dtype = torch.float
    device = torch.device("cuda:0")

    #save parameters
    # N_f = float(N) # number of test sets (float)
    # parameters={'n':n,
    #             'm_list':m_list,
    #             'N_epoch':N_epoch,
    #             'learning_rate':learning_rate,
    #             'batch_size':batch_size,
    #             'batches':batches,
    #             'test_on_new_sample':test_on_new_sample,
    #             'SGD':SGD,
    #             'gen_fun':gen_fun(-1),
    #             'K':K,
    #             'seed' : seed,
    #             'N':N,}
    # with open('./data/PARAMETERS_'+title, 'wb') as pickle_file:
    #     pickle.dump(parameters, pickle_file)

    #initialize the data and model
    if gen_fun == blob:
        print("Using blob")
        x_in = 2 # number of neurons in the input layer, i.e., dimension of data, blob:2
        H = 50
        x_out = 50 # number of neurons in the output layer (feature space)
    elif gen_fun == diffusion_cifar10:
        print("Using diffusion_cifar10")
        x_in = 3*64*64
        x_out = 300
    else:
        raise ValueError('gen_fun not supported')

    for i in range(len(n_list)):
        n = n_list[i]
        m = m_list[i]
        print("----- Starting n=%d and m=%d -----"%(n, m))
        print("----- True n=%d and m=%d -----"%(n_list_rounded[i], m_list_rounded[i]))

        print("----- Starting N_epoch=%d epochs -----"%(N_epoch))
        print("----- K=%d big trials, N=%d tests per trial for inference of Z. -----"%(K,N))
        Results = np.zeros([2, K])
        J_star_u = np.zeros([K, N_epoch])

        if not SGD:
            batch_size=n
            batches=1
            batch_size, batch_m = n, m
        else:
            batches=n//batch_size
            n=batches*batch_size #round up
            m=(m//batches)*batches
            batch_m=m//batches

        for kk in range(K):
            print("### Starting %d of %d ###"%(kk,K))

            # Generate data
            X, Y = gen_fun(n)
            Z, _ = gen_fun(m)

            # Setup model, sigma and cst are trainable parameters
            sigma=torch.tensor(0.1, dtype=float).cuda() #Make sigma trainable (or not) here
            cst=torch.tensor(1.0, dtype=float).cuda()
            if gen_fun == blob:
                model_u = ModelLatentF(x_in, H, x_out).cuda()
            elif gen_fun == diffusion_cifar10:
                model_u = ConvNet_CIFAR10().cuda()
            else:
                raise ValueError('gen_fun not supported')
            # Cut data into batches
            total_S=[(X[i*batch_size:i*batch_size+batch_size], Y[i*batch_size:i*batch_size+batch_size]) 
                     for i in range(batches)] # total_S[0~batches-1][0~1]:(batch_size,2)
            total_Z=[Z[i*batch_m:i*batch_m+batch_m] for i in range(batches)]

            # Setup optimizer for training deep kernel
            optimizer_u = torch.optim.Adam(list(model_u.parameters())+[sigma]+[cst], lr=learning_rate)

            # Train deep kernel to maximize test power
            for t in range(N_epoch):
                # Compute sigma and cst
                for ind in range(batches):
                    x, y=total_S[ind] #minibatches
                    z = total_Z[ind]
                    S = MatConvert(np.concatenate(([x, y, z]), axis=0), device, dtype)
                    Fea = model_u(S)
                    # Note: MMD_LFI_STAT takes (ϕ(x), ϕ(y), ϕ(z))
                    #       MMD_STAT     takes (ϕ(x), ϕ(y))
                    #if LfI:
                    mmd_squared_temp, mmd_squared_var_temp = MMD_LFI_STAT(Fea, S, batch_size, batch_m, sigma=sigma, cst=cst)
                    #else:    
                    #    mmd_squared_temp, mmd_squared_var_temp = MMD_STAT(Fea, S, batch_size, batch_m, sigma=sigma, cst=cst)
                    STAT_u = torch.sub(mmd_squared_temp, relu(mmd_squared_var_temp), alpha=1.0)
                    J_star_u[kk, t] = STAT_u.item()
                    optimizer_u.zero_grad()
                    STAT_u.backward(retain_graph=True)
                    optimizer_u.step()
                # Print MMD, std of MMD and J
                if t % print_every == 0:
                    print('Epoch:', t)
                    print("Objective: ", -1 * STAT_u.item())
            '''
            #testing overfitting
            print('TEST OVERFITTING:')            
            X1, Y1 = sample_blobs_Q(n, sigma_mx_2)
            with torch.torch.no_grad():
               Do Something Here
            '''
            H_u = np.zeros(N) 
            H_v = np.zeros(N) 
            print("Under this trained kernel, we run N = %d times LFI: "%N)
            print("start testing (n,m) = (%d,%d)"%(n,m))
            for k in range(N):    
                if test_on_new_sample:
                    X, Y = gen_fun(n)   
                Z1, Z2 = gen_fun(m)
                #t = time.time()
                sgn=1 if cst>0 else -1
                mmd_XZ = mmdG(X, Z1, model_u, n, m, sigma, cst, device, dtype)[0]
                mmd_YZ = mmdG(Y, Z1, model_u, n, m, sigma, cst, device, dtype)[0]
                H_u[k] = sgn*mmd_XZ<sgn*mmd_YZ    
                mmd_XZ = mmdG(X, Z2, model_u, n, m, sigma, cst, device, dtype)[0]
                mmd_YZ = mmdG(Y, Z2, model_u, n, m, sigma, cst, device, dtype)[0]
                H_v[k] = sgn*mmd_XZ>sgn*mmd_YZ
                #print(time.time() - t,'gen')

            Results[0, kk] = H_u.sum() / float(N)
            Results[1, kk] = H_v.sum() / float(N)
            print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~X): ", Results[0, kk])
            print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~Y): ", Results[1, kk])

        ##### Save np #####
        print("END")
        np.save('./data/LFI_'+str(n),Results) 
    ####Plotting    
    #LFI_plot(n_list_rounded, title=title)

if __name__ == "__main__":
    n_list = 10*np.array(range(12,13)) # number of samples in per mode
    m_list = 10*np.array(range(4,5))
    batch_size = 32 # if SGD=True, please make sure n%batch_size==0, (m*batches)%n==0
    random_seed = 42
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
            #np.random.shuffle(dataset_P)
            Xs = dataset_P[np.random.choice(dataset_P.shape[0], n)]
            #np.random.shuffle(dataset_Q)
            Ys = dataset_Q[np.random.choice(dataset_Q.shape[0], n)]
            return Xs, Ys
    
    # gen_fun = blob, diffusion_cifar10
    train_d(n_list, m_list, title=title, N_epoch=1, K=2, N=100,
            gen_fun=diffusion_cifar10, 
            SGD=True, batch_size=batch_size,
            seed=random_seed)

    #train_d(n_list, m_list, title=title, learning_rate=5e-4, K=100, N=1000, 
    #        N_epoch=1, print_every=100, batch_size=32, test_on_new_sample=False, 
    #        SGD=True, gen_fun=blob, seed=random_seed)
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