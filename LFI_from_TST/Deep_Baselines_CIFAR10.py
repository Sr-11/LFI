'''
Compared with the DK_TST code:
1. The C2ST, MMD-O, ME, SCF part was deleted.
2. Permutation test was deleted.
3. CIFAR10.1 dataset was deleted.
'''

import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
from utils import MatConvert, MMD_General, MMD_LFI_STAT, relu, MMD_STAT
import matplotlib.pyplot as plt
# from utils_HD import MatConvert, Pdist2, MMDu, TST_MMD_adaptive_bandwidth, TST_MMD_u, TST_ME, TST_SCF, TST_C2ST_D, TST_LCE_D

# Define the deep network for MMD-D
class Featurizer(nn.Module):
    def __init__(self):
        super(Featurizer, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=True), 
                     nn.Dropout2d(0)] #0.25
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 300))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)

        return feature


def mmdG(X, Y, model_u, n, m, sigma, cst, device, dtype):
    S = np.concatenate((X, Y), axis=0)
    S = MatConvert(S, device, dtype)
    Fea = model_u(S)
    n = X.shape[0]
    m = Y.shape[0]
    return MMD_General(Fea, n, m, S, sigma, cst)


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

def train_d(n_list, m_list, N_per=100, title='Default', learning_rate=5e-4, 
            K=15, N=1000, N_epoch=51, print_every=100, batch_size=50, 
            test_on_new_sample=True, SGD=True, LfI=True):  
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
    with open('./data/PARAMETERS_'+title, 'wb') as pickle_file:
        pickle.dump(parameters, pickle_file)

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
            print("### Start kk ###")
            if not SGD:
                batch_size=n
                batches=1
                batch_size, batch_m = n, m
            else:
                batches=n//batch_size
                n=batches*batch_size #round up
                m=(m//batches)*batches
                batch_m=m//batches
            
            X, Y = sample_blobs_Q(n, sigma_mx_2)
            Z, _ = sample_blobs_Q(m, sigma_mx_2)
            sigma=torch.tensor(0.1, dtype=float).cuda() #Make sigma trainable (or not) here
            cst=torch.tensor(1.0, dtype=float).cuda()
            total_S=[(X[i*batch_size:i*batch_size+batch_size], Y[i*batch_size:i*batch_size+batch_size]) for i in range(batches)]
            total_Z=[Z[i*batch_m:i*batch_m+batch_m] for i in range(batches)]
            model_u = ModelLatentF(x_in, H, x_out).cuda()
            
            # Setup optimizer for training deep kernel
            optimizer_u = torch.optim.Adam(list(model_u.parameters())+[sigma]+[cst], lr=learning_rate)
            # Train deep kernel to maximize test power
            for t in range(N_epoch):
                # Compute sigma and cst
                for ind in range(batches):
                    x, y=total_S[ind] #minibatches
                    z=total_Z[ind]
                    if LfI:
                        S=MatConvert(np.concatenate(([x, y, z]), axis=0), device, dtype)
                        Fea=model_u(S)
                        if LfI:
                            mmd_squared_temp, mmd_squared_var_temp=MMD_LFI_STAT(Fea, S, batch_size, batch_m, sigma=sigma, cst=cst)
                        else:    
                            mmd_squared_temp, mmd_squared_var_temp=MMD_STAT(Fea, S, batch_size, batch_m, sigma=sigma, cst=cst)
                        STAT_u = torch.sub(mmd_squared_temp, relu(mmd_squared_var_temp), alpha=1.0)
                    J_star_u[kk, t] = STAT_u.item()
                    optimizer_u.zero_grad()
                    STAT_u.backward(retain_graph=True)
                    optimizer_u.step()
                # Print MMD, std of MMD and J
                if t % print_every == 0:
                    print('Epoch:', t)
                    print("Statistic J: ", -1 * STAT_u.item())
            '''
            #testing overfitting
            print('TEST OVERFITTING:')            
            X1, Y1 = sample_blobs_Q(n, sigma_mx_2)
            with torch.torch.no_grad():
               Do Something Here
            '''
            H_u = np.zeros(N) 
            print("Under this trained kernel, we run N = %d times LFI: "%N)
            
            ##### Z~X #####
            for k in range(N):
                if test_on_new_sample:
                    X, Y = sample_blobs_Q(n, sigma_mx_2)
                Z, _ = sample_blobs_Q(m, sigma_mx_2)
                mmd_XZ = mmdG(X, Z, model_u, n, m, sigma, cst, device, dtype)[0]
                mmd_YZ = mmdG(Y, Z, model_u, n, m, sigma, cst, device, dtype)[0]
                H_u[k] = mmd_XZ<mmd_YZ    
            print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~X): ", H_u.sum()/N_f)
            Results[0, kk] = H_u.sum() / N_f

            ##### Z~Y #####
            for k in range(N):
                if test_on_new_sample:
                    X, Y = sample_blobs_Q(n, sigma_mx_2)
                _, Z = sample_blobs_Q(m, sigma_mx_2)
                mmd_XZ = mmdG(X, Z, model_u, n, m, sigma, cst, device, dtype)[0]
                mmd_YZ = mmdG(Y, Z, model_u, n, m, sigma, cst, device, dtype)[0]
                H_u[k] = mmd_XZ>mmd_YZ
            print("n, m=",str(n)+str('  ')+str(m),"--- P(success|Z~Y): ", H_u.sum()/N_f)
            Results[1, kk] = H_u.sum() / N_f
        ##### Save np #####
        print("END")
        np.save('./data/LFI_'+str(n),Results) 






# Setup seeds
os.makedirs("images", exist_ok=True)
np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True
is_cuda = True

# parameters setting
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate for C2STs")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n", type=int, default=1000, help="number of samples in one set")
opt = parser.parse_args()
print(opt)

# cuda setting
dtype = torch.float
device = torch.device("cuda:0")
cuda = True if torch.cuda.is_available() else False

# parameters setting
N_per = 100 # permutation times
alpha = 0.05 # test threshold
N1 = opt.n # number of samples in one set
K = 10 # number of trails
N = 100 # number of test sets
N_f = 100.0 # number of test sets (float)

# Loss function
adversarial_loss = torch.nn.CrossEntropyLoss()

# Naming variables
ep_OPT = np.zeros([K])
s_OPT = np.zeros([K])
s0_OPT = np.zeros([K])
Results = np.zeros([6,K])


# Configure data loader
# dataset_test[0][0].shape = torch.Size([3, 64, 64])
# Number of datapoints: 10000
# pixel value range: [-1,1]

dataset_test = datasets.CIFAR10(root='./data/cifar10', download=True,train=False,
                           transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=10000,
                                             shuffle=True, num_workers=1)

# Obtain CIFAR10 images
# data_all.shape = torch.Size([10000, 3, 64, 64])
# label_all.shape = torch.Size([10000])
for i, (imgs, Labels) in enumerate(dataloader_test):
    data_all = imgs
    label_all = Labels
Ind_all = np.arange(len(data_all))

# Loss function
adversarial_loss = torch.nn.CrossEntropyLoss()

# Repeat experiments K times (K = 10) and report average test power (rejection rate)
for kk in range(K):
    # Setup seeds
    torch.manual_seed(kk * 19 + N1)
    torch.cuda.manual_seed(kk * 19 + N1)
    np.random.seed(seed=1102 * (kk + 10) + N1)

    # Initialize deep networks for MMD-D (called featurizer), C2ST-S and C2ST-L (called discriminator)
    featurizer = Featurizer()
    
    # Initialize parameters
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
    sigma0OPT.requires_grad = True
    print(epsilonOPT.item())
    if cuda:
        featurizer.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Collect CIFAR10 images
    Ind_tr = np.random.choice(len(data_all), N1, replace=False)
    Ind_te = np.delete(Ind_all, Ind_tr)
    train_data = []
    for i in Ind_tr:
       train_data.append([data_all[i], label_all[i]])

    dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # Initialize optimizers
    optimizer_F = torch.optim.Adam(list(featurizer.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=0.0002)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------------------------------------------------------------------------------------------------
    #  Training deep networks for MMD-D (called featurizer), C2ST-S and C2ST-L (called discriminator)
    # ----------------------------------------------------------------------------------------------------
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            if True:
                ind = np.random.choice(N1, imgs.shape[0], replace=False)
                Fake_imgs = New_CIFAR_tr[ind]
                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))
                Fake_imgs = Variable(Fake_imgs.type(Tensor))
                X = torch.cat([real_imgs, Fake_imgs], 0)
                Y = torch.cat([valid, fake], 0).squeeze().long()

                # ------------------------------
                #  Train deep network for MMD-D
                # ------------------------------
                # Initialize optimizer
                optimizer_F.zero_grad()
                # Compute output of deep network
                modelu_output = featurizer(X)
                # Compute epsilon, sigma and sigma_0
                ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2
                # Compute Compute J (STAT_u)
                TEMP = MMDu(modelu_output, imgs.shape[0], X.view(X.shape[0],-1), sigma, sigma0_u, ep)
                mmd_value_temp = -1 * (TEMP[0])
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                # Compute gradient
                STAT_u.backward()
                # Update weights using gradient descent
                optimizer_F.step()

            else:
                break

    # Run two-sample test on the training set
    # Fetch training data
    s1 = data_all[Ind_tr]
    s2 = data_trans[Ind_tr_v4]
    S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
    Sv = S.view(2 * N1, -1)
    # Run two-sample test (MMD-D) on the training set
    h_u, threshold_u, mmd_value_u = TST_MMD_u(featurizer(S), N_per, N1, Sv, sigma, sigma0_u, ep, alpha, device, dtype)

    # Record best epsilon, sigma and sigma_0
    ep_OPT[kk] = ep.item()
    s_OPT[kk] = sigma.item()
    s0_OPT[kk] = sigma0_u.item()

    # Compute test power of MMD-D and baselines
    H_u = np.zeros(N)
    T_u = np.zeros(N)
    M_u = np.zeros(N)
    np.random.seed(1102)
    count_u = 0
    for k in range(N):
        # Fetch test data
        np.random.seed(seed=1102 * (k + 1) + N1)
        data_all_te = data_all[Ind_te]
        N_te = len(data_trans)-N1
        Ind_N_te = np.random.choice(len(Ind_te), N_te, replace=False)
        s1 = data_all_te[Ind_N_te]
        s2 = data_trans[Ind_te_v4]
        S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
        Sv = S.view(2 * N_te, -1)
        # MMD-D
        h_u, threshold_u, mmd_value_u = TST_MMD_u(featurizer(S), N_per, N_te, Sv, sigma, sigma0_u, ep, alpha, device, dtype)

        # Gather results
        count_u = count_u + h_u
        print("MMD-DK:", count_u)
        H_u[k] = h_u
        T_u[k] = threshold_u
        M_u[k] = mmd_value_u

    # Print test power of MMD-D and baselines
    print("Reject rate_u: ", H_u.sum() / N_f)
    Results[0, kk] = H_u.sum() / N_f
    print("Test Power of Baselines (K times): ")
    print(Results)
    print("Average Test Power of Baselines (K times): ")
    print("MMD-D: ", (Results.sum(1) / (kk + 1))[0])

np.save('./Results_CIFAR10_' + str(N1) + '_H1_MMD_D_Baselines', Results)