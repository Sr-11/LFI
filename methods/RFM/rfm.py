import numpy as np
import torch
from torch.linalg import solve
import RFM_kernels as RFM_kernels
from tqdm import tqdm
import hickle, pickle
import matplotlib.pyplot as plt

def laplace_kernel_M(pair1, pair2, bandwidth, M):
    return RFM_kernels.laplacian_M(pair1, pair2, bandwidth, M)

def get_grads(X, sol, L, P, batch_size=2, device='cpu', dtype=torch.float, verbose=False):
    K = laplace_kernel_M(X, X, L, P)

    dist = RFM_kernels.euclidean_distances_M(X, X, P, squared=False)
    dist = torch.where(dist < 1e-10, torch.zeros(1, device=device, dtype=dtype), dist)

    K = K/dist
    K[K == float("Inf")] = 0.

    a1 = sol.T
    n, d = X.shape
    n, c = a1.shape
    m, d = X.shape

    a1 = a1.reshape(n, c, 1)
    X1 = (X @ P).reshape(n, 1, d)
    step1 = a1 @ X1
    del a1, X1
    step1 = step1.reshape(-1, c*d)
    step2 = K.T @ step1
    del step1
    step2 = step2.reshape(-1, c, d)
    a2 = sol
    step3 = (a2 @ K).T
    del K, a2
    step3 = step3.reshape(m, c, 1)
    x1 = (X @ P).reshape(m, 1, d)
    step3 = step3 @ x1
    G = (step2 - step3) * -1/L # (batch_size, 1, 28)
    M = torch.sum(torch.transpose(G, 1, 2) @ G, dim=0) / len(G)
    del step3, step2, x1, G
    return M

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist
def median(X,Y,L=1):
    '''Implementation of the median heuristic. See Gretton 2012'''
    n1, d1 = X.shape
    n2, d2 = Y.shape
    assert d1 == d2, 'Dimensions of input vectors must match'
    Dxy = Pdist2(X, Y)
    mdist2 = torch.median(Dxy)
    sigma = torch.sqrt(mdist2)
    return sigma
    
def rfm(train_loader, test_loader,
        N_epoch=3, name=None, batch_size=2, reg=1e-3,
        train_acc=False, loader=True, classif=True,
        device=torch.device('cpu'), dtype=torch.float32,
        checkpoint_path=None,
        early_stopping=None,
        median_heuristic=True,
        patience=10,):

    L = 10
    if type(test_loader) == torch.utils.data.dataloader.DataLoader:
        X_test, y_test = get_data(test_loader)
    else:
        X_test, y_test = test_loader
        
    _, d = X_test.shape

    M = torch.eye(d, device = device, dtype=dtype)
    if median_heuristic:
        print("median heuristic: ", end='')
        sigma = median(list(train_loader)[0][0], list(train_loader)[0][0])
        print(sigma)
        M *= 1/sigma**2
    M_last_patience_stack = [torch.eye(d, device = device, dtype=dtype)]*patience

    MSE_list = []
    break_flag = False
    explode_flag = False
    for i in range(N_epoch):
        train_loader_iter = iter(train_loader)
        for iter_num, (X_train, y_train) in enumerate(tqdm(train_loader_iter)):
            K_train = laplace_kernel_M(X_train, X_train, L, M)
            sol = solve(K_train + reg * torch.eye(len(K_train), device = device, dtype=dtype), y_train).T # Find the inverse matrix, alpha in the paper
            M  = get_grads(X_train, sol, L, M, batch_size=batch_size, device=device, dtype=dtype, verbose=False)
            # validation
            K_test = laplace_kernel_M(X_train, X_test, L, M)
            preds = (sol @ K_test).T
            MSE_list.append(torch.mean(torch.square(preds - y_test)).item())
            count_sig_eff = torch.sum(y_test == (preds>0.5).squeeze()).item()
            print("Round " + str(i) +', iter ' +str(iter_num)+ ", MSE: ", MSE_list[-1])
            print("Round " + str(i) +', iter ' +str(iter_num)+ ", Acc: ", count_sig_eff / len(y_test))

            M_last_patience_stack.append(M.detach())
            M_last_patience_stack.pop(0)
            
            if early_stopping(MSE_list, patience):
                break_flag = True
                break
        if break_flag:
            break
    if max(MSE_list) > 10000 or count_sig_eff/len(y_test)<0.5 :
        explode_flag = True    
    if checkpoint_path is not None:
        hickle.dump(M, checkpoint_path + '/M.h')
        plt.plot(MSE_list)
        plt.savefig(checkpoint_path+'/MSE_epoch.png')
        plt.clf()

    return M_last_patience_stack[0], explode_flag

def get_data(loader):
    X = []
    y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)


    