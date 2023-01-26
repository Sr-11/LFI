from sklearn.utils import check_random_state
import numpy as np
import torch
import sys
from matplotlib import pyplot as plt
from tqdm import trange
import pickle
from IPython.display import clear_output
import os 
from utils import *
import scipy
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
device = torch.device("cuda:0")
dtype = torch.float32

# define network
H = 300
class DN(torch.nn.Module):
    def __init__(self):
        super(DN, self).__init__()
        self.restored = False
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, 1, bias=True),
        )
        self.out = torch.nn.Sigmoid()
    def forward(self, input):
        output = self.model(input)
        output = self.out(output)
        return output
    def LBI(self, x):
        return self.model(x)

# train
def train(model, total_S, total_labels, validation_S, validation_labels,
          batch_size=1024, lr=0.0002, epochs=1000, load_epoch=0, save_per=10, momentum = 0.99, n=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)    
    criterion = torch.nn.BCELoss().cuda()
    validation_records = np.ones(epochs)*np.inf
    for epoch in trange(load_epoch, epochs+load_epoch):
        order = np.random.permutation(len(total_S))
        for i in order:
            optimizer.zero_grad()
            S = total_S[i]
            labels = total_labels[i]
            outputs = model(S)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # validation
        with torch.no_grad():
            validation_records[epoch] = criterion(model(validation_S), validation_labels).detach().cpu().numpy()
        print('Epoch: %d, Loss: %.4f'%(epoch, validation_records[epoch]))
        if early_stopping(validation_records, epoch):
            break
        if epoch % save_per == 0:
            path = './checkpoint%d/'%n+str(epoch)+'/'
            try:
                os.makedirs(path) 
            except:
                pass
            torch.save(model.state_dict(), path+'model.pt')
            plt.plot(validation_records[:epoch])
            plt.savefig('./checkpoint%d/'%n+'loss.png')
            plt.clf()

    torch.save(model.state_dict(), './checkpoint%d/0/model.pt'%n)
    return model

if __name__ == "__main__":
    ##### load data #####
    dataset = np.load('../HIGGS.npy')
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28), 0
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28), 1
    del dataset

    n_list = [1300000, 1000000, 700000, 400000, 200000, 50000]
    repeats = 10
    # for n in [1300000, 1000000, 700000, 400000, 200000, 50000]:
    #     for i in range(11):
    #         n_list.append(n+i)
    
    for n in n_list:
        X, Y = dataset_P[:n], dataset_Q[:n]
        batch_size = 1024
        batches = n//batch_size
        
        #### training set ####
        total_S = [(X[i*batch_size:(i+1)*batch_size], 
                    Y[i*batch_size:(i+1)*batch_size]) 
                    for i in range(batches)]
        total_S = [MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in total_S]
        total_labels = [torch.cat((torch.zeros((batch_size,1), dtype=dtype), 
                                torch.ones((batch_size,1), dtype=dtype))
                                ).to(device) for _ in range(batches)]
        ##### Validation #####
        validation_S = MatConvert(np.concatenate((dataset_P[n:n+10000], dataset_Q[n:n+10000]), axis=0), device, dtype)
        validation_labels = torch.cat((torch.zeros((10000,1), dtype=dtype),
                                    torch.ones((10000,1), dtype=dtype))
                                    ).to(device)
        ##### Train #####
        P_values = np.zeros(repeats)
        for i in range(repeats):
            n_train = n
            model = DN().to(device)
            model = train(model, total_S, total_labels, validation_S, validation_labels,
                        epochs=501, 
                        batch_size=batch_size, lr=2e-3, momentum=0.99,
                        load_epoch=0, save_per=10, 
                        n=n_train+i)

        ### Evaluation ###
            n_eval = 10000
            X_eval = dataset_P[n_train:n_train+n_eval]
            Y_eval = dataset_Q[n_train:n_train+n_eval]
            X_eval = MatConvert(X_eval, device, dtype)
            Y_eval = MatConvert(Y_eval, device, dtype)

            n_test = 10000
            X_test = dataset_P[n_train+n_eval:n_train+n_eval+n_test]
            Y_test = dataset_Q[n_train+n_eval:n_train+n_eval+n_test]
            X_test = MatConvert(X_test, device, dtype)
            Y_test = MatConvert(Y_test, device, dtype)

            pval = get_pval_at_once(X_eval, Y_eval, X_eval, Y_eval, X_test, Y_test,
                                model, None, 'LBI', None, None, None,
                                norm_or_binom = False)
            P_values[i] = pval
            print('P-value: %.4f'%pval)
        np.save('./checkpoint%d/P_values.npy'%n_train, P_values)

        