import numpy as np
import torch
import sys
from sklearn.utils import check_random_state
from matplotlib import pyplot as plt
from tqdm import trange
import torchvision.datasets as datasets
#### For gen_fun inputs, make sure they take in n and return X and Y
#### Moreover, make sure when n=-1 they return a string for the title

def sample_blobs_Q(N1, sigma_mx_2, rows=3, cols=3, rs=None):
    """Generate Blob-D for testing type-II error (or test power)."""
    """ X ~ N(0, 0.03) + randint """
    """ Y ~ N(0, sigma_mx_2(9*2*2)) + {0,1,2}^2 """
    rs = check_random_state(rs)
    mu = np.zeros(2)
    sigma = np.eye(2) * 0.03
    X = rs.multivariate_normal(mu, sigma, size=N1)
    Y = rs.multivariate_normal(mu, np.eye(2), size=N1)
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

def blob(n):
    """ input: n, number of samples """
    """ output: (n,d) numpy array, d is dimension of a datapoint """
    if n <0 :
        return 'BLOB'
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
    return sample_blobs_Q(n, sigma_mx_2)

def diffusion_cifar10(n):
    '''
    To avoid loading the data multiple times, please use diffusion_cifar10() in LFI.py
    '''
    return None

    if n <0 :
        return 'DIFFUSION'

    diffusion = np.load("./Diffusion/ddpm_generated_images.npy")
    np.random.shuffle(diffusion)
    diffusion = diffusion.reshape(n, -1)

    try:
        trainset = datasets.CIFAR10(root='./data', train=True, download=False)
        testset = datasets.CIFAR10(root='./data', train=False, download=False)
    except:
        trainset = datasets.CIFAR10(root='./data', train=True, download=True)
        testset = datasets.CIFAR10(root='./data', train=False, download=True)

    cifar10 = np.zeros((n,32,32,3))
    for i in range(n):
        j = np.random.randint(0,50000)
        cifar10[i] = np.asarray(trainset[j][0])
    cifar10 = cifar10.reshape(n, -1)
    return diffusion, cifar10