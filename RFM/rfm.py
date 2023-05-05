import numpy as np
import torch
from torch.linalg import solve
import kernels
from tqdm import tqdm
import hickle, pickle
import matplotlib.pyplot as plt

def laplace_kernel_M(pair1, pair2, bandwidth, M):
    return kernels.laplacian_M(pair1, pair2, bandwidth, M)


def get_grads(X, sol, L, P, batch_size=2, device='cpu', verbose=False):
    M = 0.

    num_samples = 20000
    indices = np.random.randint(len(X), size=num_samples)

    if len(X) > len(indices):
        x = X[indices, :]
    else:
        x = X

    K = laplace_kernel_M(X, x, L, P)

    dist = kernels.euclidean_distances_M(X, x, P, squared=False)
    dist = torch.where(dist < 1e-10, torch.zeros(1, device=device), dist)

    K = K/dist
    K[K == float("Inf")] = 0.

    a1 = sol.T
    n, d = X.shape
    n, c = a1.shape
    m, d = x.shape

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
    x1 = (x @ P).reshape(m, 1, d)
    step3 = step3 @ x1

    G = (step2 - step3) * -1/L

    M = 0.

    bs = batch_size
    batches = torch.split(G, bs)
    if verbose:
        for i in tqdm(range(len(batches))):
            grad = batches[i]
            gradT = torch.transpose(grad, 1, 2)
            M += torch.sum(gradT @ grad, dim=0)
            del grad, gradT
    else:
        for i in range(len(batches)):
            grad = batches[i]
            gradT = torch.transpose(grad, 1, 2)
            M += torch.sum(gradT @ grad, dim=0)
            del grad, gradT
    M /= len(G)
    # M = M.numpy()

    return M


def rfm(train_loader, test_loader,
        iters=3, name=None, batch_size=2, reg=1e-3,
        train_acc=False, loader=True, classif=True,
        device=torch.device('cpu'),
        checkpoint_path=None,
        early_stopping=None,):

    L = 10
    
    if loader:
        print("Loaders provided")
        # X_train, y_train = get_data(train_loader)
        X_test, y_test = get_data(test_loader)
    else:
        # X_train, y_train = train_loader
        X_test, y_test = test_loader
        # X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()
        # y_train = torch.from_numpy(y_train).float()
        y_test = torch.from_numpy(y_test).float()
        
    _, d = X_test.shape

    M = torch.eye(d, device = device)
    
    MSE_list = []
    break_flag = False
    for i in range(iters):
        train_loader_iter = iter(train_loader)
        for X_train, y_train in tqdm(train_loader_iter):
            K_train = laplace_kernel_M(X_train, X_train, L, M)
            sol = solve(K_train + reg * torch.eye(len(K_train), device = device), y_train).T # Find the inverse matrix, alpha in the paper
            M  = get_grads(X_train, sol, L, M, batch_size=batch_size, device=device, verbose=False)
            # validation
            K_test = laplace_kernel_M(X_train, X_test, L, M)
            preds = (sol @ K_test).T
            MSE_list.append(torch.mean(torch.square(preds - y_test)).item())
            count = torch.sum(y_test == (preds>0.5)).item()

            print("Round " + str(i) + " MSE: ", torch.mean(torch.square(preds - y_test)))
            print("Round " + str(i) + " Acc: ", count / len(y_test))

            # if train_acc:
            #     preds = (sol @ K_train).T
            #     y_pred = torch.from_numpy(preds)
            #     preds = torch.argmax(y_pred, dim=-1)
            #     labels = torch.argmax(y_train, dim=-1)
            #     count = torch.sum(labels == preds).numpy()
            #     print("Round " + str(i) + " Train Acc: ", count / len(labels))
            # if classif:
            #     y_pred = preds
            #     preds = torch.argmax(y_pred, dim=-1)
            #     labels = torch.argmax(y_test, dim=-1)
            #     count = torch.sum(labels == preds)
            #     print("Round " + str(i) + " Test Acc: ", count / len(labels))

            if early_stopping(MSE_list,len(MSE_list)):
                break_flag = True
                break
        if break_flag:
            break
        
    if checkpoint_path is not None:
        hickle.dump(M, checkpoint_path + '/M.h')
        plt.plot(MSE_list)
        plt.savefig(checkpoint_path+'/MSE_epoch.png')
        plt.clf()

    return M

def get_data(loader):
    X = []
    y = []
    for idx, batch in enumerate(loader):
        inputs, labels = batch
        X.append(inputs)
        y.append(labels)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)


       

    # K_train = laplace_kernel_M(X_train, X_train, L, M)
    # sol = solve(K_train + reg * torch.eye(len(K_train), device = device), y_train).T
    # K_test = laplace_kernel_M(X_train, X_test, L, M)
    # preds = (sol @ K_test).T
    # mse = torch.mean(torch.square(preds - y_test))
    # print("Final MSE: ", mse)
    
    # if classif:
    #     y_pred = torch.from_numpy(preds)
    #     preds = torch.argmax(y_pred, dim=-1)
    #     labels = torch.argmax(y_test, dim=-1)
    #     count = torch.sum(labels == preds)
    #     print(" Final Acc: ", count / len(labels))
        
