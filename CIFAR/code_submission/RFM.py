import numpy as np
import torch
import sys
from sklearn.utils import check_random_state
from utils import *
from matplotlib import pyplot as plt
import pickle
import torch.nn as nn
device = torch.device("cuda:0")
dtype = torch.float32
def load_diffusion_cifar_32():
    diffusion = np.load("../../Diffusion/ddpm_generated_images2.npy").transpose(0,3,1,2)
    cifar10 = np.load('../../data/cifar_data.npy')
    dataset_P = diffusion.reshape(diffusion.shape[0], -1)
    dataset_Q = cifar10.reshape(cifar10.shape[0], -1)
    return dataset_P, dataset_Q[:10000, :], dataset_Q[10000:, :]

DP, DQ_1, DQ_2 = load_diffusion_cifar_32()
#print(DP.shape, DQ_1.shape, DQ_2.shape): (13140, 3072) (10000, 3072) (40000, 3072)
mix_rate=2 #For each DP, match with mix_rate*DQ data points
test_DP1=np.concatenate((DP[:2000, :], DQ_1[:4000, :]), axis=0)
test_DQ1=DQ_1[4000: 10000, :]
train_DP1=np.concatenate((DP[2000:7000, :], DQ_2[:10000, :]), axis=0)
train_DQ1=DQ_2[10000: 25000, :]
#generate a random shuffle over train_DP1, print the first item
train_DP1 = train_DP1[np.random.choice(train_DP1.shape[0], train_DP1.shape[0], replace=False), :]
train_DQ1 = train_DQ1[np.random.choice(train_DQ1.shape[0], train_DQ1.shape[0], replace=False), :]
test_DP1 = test_DP1[np.random.choice(test_DP1.shape[0], test_DP1.shape[0], replace=False), :]
test_DQ1 = test_DQ1[np.random.choice(test_DQ1.shape[0], test_DQ1.shape[0], replace=False), :]
def gen_fun1(n): #n at most 15000
    X = train_DP1[np.random.choice(train_DP1.shape[0], n, replace=False), :]
    Y = train_DQ1[np.random.choice(train_DQ1.shape[0], n, replace=False), :]
    return X, Y
def gen_fun2(n): #n at most 6000
    X = test_DP1[np.random.choice(test_DP1.shape[0], n, replace=False), :]
    Y = test_DQ1[np.random.choice(test_DQ1.shape[0], n, replace=False), :]
    return X, Y
def euclidean_distances_M(samples, centers, M, squared=True):
    samples_norm = (samples @ M)  * samples
    samples_norm = np.sum(samples_norm, axis=1, keepdims=True)
    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = (centers @ M) * centers
        centers_norm = np.sum(centers_norm, axis=1, keepdims=True)
    centers_norm = np.reshape(centers_norm, (1, -1))
    distances = np.matmul(samples, M @ np.transpose(centers))
    distances*=(-2.0)
    distances+=samples_norm
    distances+=centers_norm
    np.clip(distances, a_min=0, a_max=None, out=distances)
    if not squared:
        distances=np.sqrt(distances)
    return distances
def laplace_kernel(samples, centers, M):
    kernel_mat = euclidean_distances_M(samples, centers, M, squared=False)
    np.clip(kernel_mat, a_min=0, a_max=None, out=kernel_mat)
    kernel_mat*=(-1.0)
    kernel_mat=np.exp(kernel_mat)
    return kernel_mat
def mmd_rfm(X, Y, M):
    Kx = laplace_kernel(X, X, M)
    Ky = laplace_kernel(Y, Y, M)
    Kxy = laplace_kernel(X, Y, M)
    return mmd(Kx, Kxy, Ky)
def get_grads(X, sol, P):
    K = laplace_kernel(X, X, P)
    dist = euclidean_distances_M(X, X, P, squared=False)
    K = K/dist
    K[dist == 0.] = 0.
    a1 = sol.T
    n, d = X.shape
    n, c = a1.shape
    a1 = a1.reshape(n, c, 1)
    X1 = (X @ P).reshape(n, 1, d)
    step1 = a1 @ X1
    step1 = step1.reshape(-1, c*d)
    step2 = K.T @ step1
    step2 = step2.reshape(-1, c, d)
    a2 = sol
    step3 = (a2 @ K).T
    step3 = step3.reshape(n, c, 1)
    x1 = (X @ P).reshape(n, 1, d)
    step3 = step3 @ x1
    G = (step2 - step3) * -1.0
    G = G.reshape(-1, c*d)
    M = np.zeros((d, d))
    for i in range(len(G)):
        M += np.outer(G[i], G[i])
    M /= len(G)
    return M
if __name__ == '__main__':
    n=1920
    N=300
    d=3072
    reg = 1e-3
    m_list=[48, 96, 128, 192, 256, 384]
    for _  in range(5): 
            X, Y = gen_fun1(n)
            X, Y = X/256.0, Y/256.0
            batch_size=64
            M = np.identity(d)
            for _ in range(50):
                sample = np.random.choice(len(X), batch_size, replace=False)
                X_1, Y_1 = X[sample], Y[sample]
                X_train = np.concatenate((X_1, Y_1), axis=0)
                y_train = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))), axis=0)
                permute = np.random.permutation(len(X_train))
                X_train, y_train = X_train[permute], y_train[permute]
                K_train = laplace_kernel(X_train, X_train, M)
                sol = np.linalg.solve(K_train + reg * np.eye(len(K_train)), y_train).T
                M  = get_grads(X_train, sol, M)
            H_u = np.zeros(N) 
            H_v = np.zeros(N)
            R_u = np.zeros(N)
            R_v = np.zeros(N)
            P_u = np.zeros(N)
            P_v = np.zeros(N)
            print("Under this trained kernel, we run N = %d times LFI: "%N)
            for i in range(len(m_list)):
                stat = []
                print("start testing m = %d"%m_list[i])
                m = m_list[i]
                for j in range(500):
                        Z_temp = gen_fun2(m)[0]/256.0
                        mmd_XZ = mmd_rfm(X, Z_temp, M)
                        mmd_YZ = mmd_rfm(Y, Z_temp, M)
                        stat.append(float(mmd_XZ - mmd_YZ))
                stat = np.sort(stat)
                thres = np.percentile(stat, 95)
                for k in range(N):     
                    if k%25==0:
                        print("start testing %d-th data trial"%k)
                    Z1, Z2 = gen_fun2(m)
                    Z1, Z2 = Z1/256.0, Z2/256.0
                    mmd_XZ = mmd_rfm(X, Z1, M)
                    mmd_YZ = mmd_rfm(Y, Z1, M)
                    H_u[k] = mmd_XZ - mmd_YZ < 0.0
                    R_u[k] = mmd_XZ - mmd_YZ < thres
                    P_u[k] = np.searchsorted(stat, float(mmd_XZ - mmd_YZ), side="left")/500.0
                    mmd_XZ = mmd_rfm(X, Z2, M)
                    mmd_YZ = mmd_rfm(Y, Z2, M)
                    H_v[k] = mmd_XZ - mmd_YZ > 0.0
                    R_v[k] = mmd_XZ - mmd_YZ > thres
                    P_v[k] = np.searchsorted(stat, float(mmd_XZ - mmd_YZ), side="left")/500.0
                print("n, m=",str(n)+str('  ')+str(m),"--- P(T0|Z~X): ", H_u.mean())
                print("n, m=",str(n)+str('  ')+str(m),"--- P(T0|Z~Y): ", H_v.mean())
                print("n, m=",str(n)+str('  ')+str(m),"--- P(95|Z~X): ", R_u.mean())
                print("n, m=",str(n)+str('  ')+str(m),"--- P(95|Z~Y): ", R_v.mean())
                print("n, m=",str(n)+str('  ')+str(m),"--- P(Ep|Z~X): ", P_u.mean())
                print("n, m=",str(n)+str('  ')+str(m),"--- P(Ep|Z~Y): ", P_v.mean())