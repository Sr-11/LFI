from sklearn.utils import check_random_state
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import trange
from utils import *
from matplotlib import pyplot as plt
import torch.nn as nn
dtype = torch.float
device = torch.device("cuda")
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
criterion = torch.nn.CrossEntropyLoss().cuda()
def load_diffusion_cifar_32():
    diffusion = np.load("../Diffusion/ddpm_generated_images2.npy").transpose(0,3,1,2)
    cifar10 = np.load('../data/cifar_data.npy')
    dataset_P = diffusion.reshape(diffusion.shape[0], -1)
    dataset_Q = cifar10.reshape(cifar10.shape[0], -1)
    return dataset_P, dataset_Q

DP, DQ = load_diffusion_cifar_32()
DP1 = DP[:4000, :]
DQ1 = DQ[:4000, :]
DP2 = DP[4000:13140, :]
DQ2 = DQ[4000:13140, :]
DP1_T = torch.tensor(DP1, dtype=dtype, device=device)
DQ1_T = torch.tensor(DQ1, dtype=dtype, device=device)
DP2_T = torch.tensor(DP2, dtype=dtype, device=device)
DQ2_T = torch.tensor(DQ2, dtype=dtype, device=device)
def gen_fun1(n):
    X = DP1[np.random.choice(DP1.shape[0], n, replace=False), :]
    Y = DQ1[np.random.choice(DQ1.shape[0], n, replace=False), :]
    return X, Y
def gen_fun2(n):
    X = DP2[np.random.choice(DP2.shape[0], n, replace=False), :]
    Y = DQ2[np.random.choice(DQ2.shape[0], n, replace=False), :]
    return X, Y
def gen_fun1_T(n):
    X = DP1_T[np.random.choice(DP1_T.shape[0], n, replace=False), :]
    Y = DQ1_T[np.random.choice(DQ1_T.shape[0], n, replace=False), :]
    return X, Y
def gen_fun2_T(n):
    X = DP2_T[np.random.choice(DP2_T.shape[0], n, replace=False), :]
    Y = DQ2_T[np.random.choice(DQ2_T.shape[0], n, replace=False), :]
    return X, Y
class ConvNet_CIFAR10(nn.Module):
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
            nn.Unflatten(1,(3, 32,32)),
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128))
        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4 # for 64*64 image, set ds_size = 64 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 300),
                                    nn.ReLU(),  
                                    nn.Linear(300, 2)) 
    def forward(self, img):
        out = self.model(img).to(device)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out).to(device)
        return feature



def train(X, Y, criterion, batch_size=32, lr=0.0002, epochs=100):
    """Label the items first
        items in X have label 0, items in Y have label 1
        then train the model with sgd
        X, Y are numpy arrays
        X has shape (N1, 2), Y has shape (N2, 2)
    """
    model=ConvNet_CIFAR10().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_records = []
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            optimizer.zero_grad()
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            X_batch = torch.from_numpy(X_batch).float()
            Y_batch = torch.from_numpy(Y_batch).float()
            output_X = model(X_batch.to(device)).to(device)
            output_Y = model(Y_batch.to(device)).to(device)
            loss_X = criterion(output_X, torch.zeros(len(X_batch)).long().to(device))
            loss_Y = criterion(output_Y, torch.ones(len(Y_batch)).long().to(device))
            loss = loss_X + loss_Y
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            loss_records.append(loss.detach().cpu().numpy())
            print("epoch: ", epoch)
            print("loss: ", loss)
            print("accuracy on X: ", Scheffe(model, X))
            print("accuracy on Y: ", 1-Scheffe(model, Y))
            X1, Y1 = gen_fun2(1000)
            print("accuracy on X_test: ", Scheffe(model, X1))
            print("accuracy on Y_test: ", 1-Scheffe(model, Y1))
    return model

def test(model, X, Y, verbose=False):
    '''
    X has label 0, Y has label 1
    Test accuracy of model on X and Y
    '''
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    class_X = torch.argmax(model(X.to(device)), dim=1)
    class_Y = torch.argmax(model(Y.to(device)), dim=1)
    print("accuracy on X: ", float(torch.sum(class_X == 0).float() / len(X)))
    print("accuracy on Y: ", float(torch.sum(class_Y == 1).float() / len(Y)))

def likelihood(model, Z):
    '''
    Do the following:
    1. Apply model to Z
    2. Compute softmax of output
    3. Return probability of class 0
    '''
    Z = torch.from_numpy(Z).float()
    output_Z = model(Z.to(device)) #has shape (batch_size, 2)
    prob = torch.softmax(output_Z, dim=1)[:,0] #Probability of class 0
    return np.mean(prob.detach().cpu().numpy())
def Scheffe(model,Z):
    '''
    Output number of Z's classified as class 0
    '''
    with torch.no_grad():
        success = 0
        Z = torch.from_numpy(Z).float()
        class_ = torch.argmax(model(Z.to(device)), dim=1)
        success = torch.sum(class_ == 0).float() / len(Z)
        return float(success)
def Scheffe_T(model, Z_T):
    '''
    Output number of Z's classified as class 0 but Z is already a tensor
    '''
    with torch.no_grad():
        success = 0
        class_ = torch.argmax(model(Z_T.to(device)), dim=1)
        success = torch.sum(class_ == 0).float() / len(Z_T)
        return float(success)
def likelihood_T(model, Z_T):
    '''
    Output number of Z's classified as class 0 but Z is already a tensor
    '''
    with torch.no_grad():
        output_Z = model(Z_T.to(device)) #has shape (batch_size, 2)
        prob = torch.softmax(output_Z, dim=1)[:,0] #Probability of class 0
        return np.mean(prob.detach().cpu().numpy())
    

n=256
m_list = list(range(4,64, 4))
for k in range(10):
        F1 = './data/RRresult_SCH_'+str(n)+'_'+str(k)+'_.txt'
        F2 = './data/RRresult_LBI_'+str(n)+'_'+str(k)+'_.txt'

        fwrite('', F1, message='New File '+str(n))
        fwrite('', F2, message='New File '+str(n))
        if True:
            X, Y = gen_fun1(n)
            model=train(X, Y, criterion, batch_size=32, lr=0.0001, epochs=100)

            for m in m_list:
                H=[]
                for i in range(1000):
                    X1, Y1 = gen_fun1(m)
                    H.append(Scheffe(model, X1))
                stat = np.sort(H)
                thres = stat[49]
                #thres = np.percentile(H, 5)
                H=[[], []]
                T=[[], []]
                P=[[], []]
                for i in range(1000):
                    X1, Y1 = gen_fun1(m)
                    tx = Scheffe(model, X1)
                    ty = Scheffe(model, Y1)
                    H[0].append(tx>=thres)
                    H[1].append(ty<thres)
                    T[0].append(tx>0.5)
                    T[1].append(ty<0.5)
                    P[0].append(np.searchsorted(stat, tx)/1000.0)
                    P[1].append(np.searchsorted(stat, ty)/1000.0)
                st1="n, m= "+str(n)+str('  ')+str(m)+" --- P(max|Z~X):  "+str(np.mean(T[0]))
                st2="n, m= "+str(n)+str('  ')+str(m)+" --- P(max|Z~Y):  "+str(np.mean(T[1]))
                st3="n, m= "+str(n)+str('  ')+str(m)+" --- P(95|Z~X):  "+str(np.mean(H[0]))
                st4="n, m= "+str(n)+str('  ')+str(m)+" --- P(95|Z~Y):  "+str(np.mean(H[1]))
                st5="n, m= "+str(n)+str('  ')+str(m)+" --- P(p|Z~X):  "+str(1-np.mean(P[0]))
                st6="n, m= "+str(n)+str('  ')+str(m)+" --- P(p|Z~Y):  "+str(1-np.mean(P[1]))
                fwrite(st1, F1)
                fwrite(st2, F1)
                fwrite(st3, F1)
                fwrite(st4, F1)
                fwrite(st5, F1)
                fwrite(st6, F1)
            print('Done with Scheffe')
            for m in m_list:
                H=[]
                for i in range(1000):
                    X1, Y1 = gen_fun1(m)
                    H.append(likelihood(model, X1))
                #thres = np.percentile(H, 5)
                stat = np.sort(H)
                thres = stat[49]
                H=[[], []]
                T=[[], []]
                P=[[], []]
                for i in range(1000):
                    X1, Y1 = gen_fun1(m)
                    tx = likelihood(model, X1)
                    ty = likelihood(model, Y1)
                    H[0].append(tx>=thres)
                    H[1].append(ty<thres)
                    T[0].append(tx>0.5)
                    T[1].append(ty<0.5)
                    P[0].append(np.searchsorted(stat, tx)/1000.0)
                    P[1].append(np.searchsorted(stat, ty)/1000.0)                
                st1="n, m= "+str(n)+str('  ')+str(m)+" --- P(max|Z~X):  "+str(np.mean(T[0]))
                st2="n, m= "+str(n)+str('  ')+str(m)+" --- P(max|Z~Y):  "+str(np.mean(T[1]))
                st3="n, m= "+str(n)+str('  ')+str(m)+" --- P(95|Z~X):  "+str(np.mean(H[0]))
                st4="n, m= "+str(n)+str('  ')+str(m)+" --- P(95|Z~Y):  "+str(np.mean(H[1]))
                st5="n, m= "+str(n)+str('  ')+str(m)+" --- P(p|Z~X):  "+str(1-np.mean(P[0]))
                st6="n, m= "+str(n)+str('  ')+str(m)+" --- P(p|Z~Y):  "+str(1-np.mean(P[1]))
                fwrite(st1, F2)
                fwrite(st2, F2)
                fwrite(st3, F2)
                fwrite(st4, F2)
                fwrite(st5, F2)
                fwrite(st6, F2)
            print('Done with likelihood')