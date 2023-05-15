import numpy as np
import torch
from UME_utils import *
from matplotlib import pyplot as plt
import os
from tqdm import tqdm, trange
# import autograd.numpy as np
import pickle
import sys
import UME_config as UME_config

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= UME_config.train_param_configs['gpu_id'] # specify which GPU(s) to be used
 
torch.backends.cudnn.deterministic = True
dtype = torch.float
device = torch.device("cuda:0")
torch.manual_seed(42)
np.random.seed(42)

debug = False

# define the loss function
def power_criterion_mix(XY, V, kernel): #  objective to maximize, 
    """
    Note: compared to Jit18, here R is linear mixture of P and Q, we don't need R for the objective
    """
    # compute the mean and variance of the test statistic
    TEMP = kernel.compute_UME_mean_variance(XY, V)
    # calculate objective
    UME_mean = TEMP[0]
    UME_var = TEMP[1]
    UME_std = torch.sqrt(UME_var+10**(-6))
    # print(UME_mean, UME_std)
    ratio = torch.div(UME_mean,UME_std)
    return ratio

# define the early stopping criterion
def early_stopping(validation_losses, epoch):
    i = np.argmin(validation_losses)
    # print(i)
    if epoch - i > 10 and validation_losses[i]<-0.1:
        return True
    else:
        return False

# main training function
def optimize_3sample_criterion_and_kernel(prepared_batched_XY, # total_S, is a list
                                          validation_XY, # validation set, is 2n*d
                                        V, kernel, # params
                                        N_epoch, learning_rate, momentum, # optimizer
                                        print_every, # print and save checkpoint
                                        early_stopping, # early stopping boolean function
                                        fig_loss_epoch_path, # name of the loss vs epoch figure
                                        chechpoint_folder_path # folder path to save checkpoint
                                        ):
    params = kernel.params #+ [V]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum)
    total_S = prepared_batched_XY
    batches = len(total_S)
    #############################
    # start training
    #############################
    validation_ratio_list = np.ones([N_epoch])*np.inf
    for t in range(N_epoch):
        print('epoch',t)
        order = np.random.permutation(batches)
        for ind in tqdm(order):
            optimizer.zero_grad()
            # calculate parameters
            XY = total_S[ind]
            # calculate MMD
            ratio = power_criterion_mix(XY, V, kernel)
            obj = -ratio
            # update parameters
            obj.backward(retain_graph=False)
            optimizer.step()     
            # prevent nan
            kernel.clamp()
        #validation
        with torch.torch.no_grad():
            validation_ratio = -power_criterion_mix(validation_XY, V, kernel).item()
            validation_ratio_list[t] = validation_ratio
            print('validation =', validation_ratio_list[t])
        # print log
        if t%print_every==0:
            # print(validation_ratio_list[:t])
            plt.plot(validation_ratio_list[:t])
            plt.savefig(fig_loss_epoch_path)
            plt.clf()
            save_model(V, kernel, t, chechpoint_folder_path)
        # early stopping
        if early_stopping(validation_ratio_list, t):
            save_model(V, kernel, t, chechpoint_folder_path)
            plt.plot(validation_ratio_list[:t])
            plt.savefig(fig_loss_epoch_path)
            plt.clf()
            print('Best epoch:', np.argmin(validation_ratio_list))
            with open(chechpoint_folder_path+'data.pickle', 'wb') as f:
                pickle.dump({'epoch': t}, f)
            return V, kernel
    return V, kernel

# further encapsulate the training function
def train(n_tr, J=None, # size of X_tr, Y_tr and W=V
        load_epoch=None, # load checkpoint if >0
        batch_size=None, N_epoch=None, learning_rate=None, momentum=None, # optimizer
        print_every=None, # print and save checkpoint
        early_stopping=None, # early stopping boolean function
        fig_loss_epoch_path=None, # name of the loss vs epoch figure
        chechpoint_folder_path=None, # folder path to save checkpoint
        ):
    n_backup = n_tr
    try:
        os.mkdir(chechpoint_folder_path)
    except:
        pass

    batches = n_tr//batch_size + 1 
    n = batches*batch_size  
    X = dataset_P[0:n]
    Y = dataset_Q[0:n]

    # prepare training data
    total_S = [(X[i*batch_size:(i+1)*batch_size], 
                Y[i*batch_size:(i+1)*batch_size]) 
                for i in range(batches)]
    total_S = [MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in total_S]

    # prepare NN and kernel and V=W 
    model = DN().cuda()
    another_model = another_DN().cuda()

    epsilonOPT = MatConvert(np.zeros(1), device, dtype) # set to 0 for MMD-G
    epsilonOPT.requires_grad = True
    sigmaOPT = 10*MatConvert(np.sqrt(np.random.rand(1) * 0.3), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = 10*MatConvert(np.sqrt(np.random.rand(1) * 0.002), device, dtype)
    sigma0OPT.requires_grad = True
    eps=MatConvert(np.zeros((1,)), device, dtype)
    eps.requires_grad = True
    cst=MatConvert(1*np.ones((1,)), device, dtype)
    cst.requires_grad = False
    kernel = NeuralKernel(model, another_model, epsilonOPT, sigmaOPT, sigma0OPT, eps, cst)

    V = np.concatenate((dataset_P[np.random.choice(n, J//2, replace=False)], 
                        dataset_Q[np.random.choice(n, J-J//2, replace=False)]), axis=0)
    V = MatConvert(V, device, dtype)

    # load checkpoint if one want to start from a previous checkpoint
    if load_epoch>0:
        V, kernel = load_model(chechpoint_folder_path, epoch=0)
        print('loaded')
    kernel.model.eval()
    kernel.another_model.eval()

    # validation data
    validation_XY = np.concatenate((dataset_P[ n: n+10000 ], 
                            dataset_Q[ n: n+10000 ]), axis=0)
    validation_XY = MatConvert(validation_XY, device, dtype)

    #############################
    # start training
    #############################
    return optimize_3sample_criterion_and_kernel(total_S, # total_S, is a list
                                          validation_XY, # validation set, is 2n*d
                                        V, kernel, # params
                                        N_epoch=N_epoch, learning_rate=learning_rate, momentum=momentum, # optimizer
                                        print_every=print_every, # print and save checkpoint
                                        early_stopping=early_stopping, # early stopping boolean function
                                        fig_loss_epoch_path=fig_loss_epoch_path, # name of the loss vs epoch figure
                                        chechpoint_folder_path = chechpoint_folder_path, # folder path to save checkpoint
                                        )


if __name__ == "__main__":
    dataset = np.load(UME_config.resource_configs['Higgs_path'])
    print('signal : background =',np.sum(dataset[:,0]),':',dataset.shape[0]-np.sum(dataset[:,0]))
    print('signal :',np.sum(dataset[:,0])/dataset.shape[0]*100,'%')
    # split into signal and background
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5170877, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5829122, 28) 

    n_org_list = UME_config.train_param_configs['n_tr_list']
    repeats = UME_config.train_param_configs['repeats']
    n_tr_list = []
    for n in n_org_list:
        for i in range(repeats):
            n_tr_list.append(n+i)
    
    J_tr = UME_config.train_param_configs['J_tr']
    batch_size = UME_config.train_param_configs['batch_size']
    N_epoch = UME_config.train_param_configs['N_epoch']
    learning_rate = UME_config.train_param_configs['learning_rate']
    momentum = UME_config.train_param_configs['momentum']
    checkpoint_path = UME_config.expr_configs['checkpoints_path']

    for n_tr in n_tr_list:
        print('------ n =', n_tr, '------')
        V, kernel =  train(n_tr, J=J_tr, # size of X_tr, Y_tr and W=V
            load_epoch=0, # load checkpoint if >0
            batch_size=batch_size, N_epoch=N_epoch, learning_rate=learning_rate, momentum=momentum, # optimizer
            print_every=10, # print and save checkpoint
            early_stopping=early_stopping, # early stopping boolean function
            fig_loss_epoch_path=checkpoint_path+'/checkpoints n_tr=%d/loss_epoch.png'%n_tr, # loss vs epoch figure
            chechpoint_folder_path = checkpoint_path+'/checkpoints n_tr=%d/'%n_tr, # folder path to save checkpoint
            )

