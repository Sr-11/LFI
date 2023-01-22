#Implements Scheffes test by first building a classifier between X and Y, then classifies Y.
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
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used
H = 300
class DN_theirs(torch.nn.Module):
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
            torch.nn.Sigmoid()
        )
        torch.nn.init.normal_(self.model[0].weight,0,0.1)
        torch.nn.init.normal_(self.model[2].weight,0,0.05)
        torch.nn.init.normal_(self.model[4].weight,0,0.05)
        torch.nn.init.normal_(self.model[6].weight,0,0.05)
        torch.nn.init.normal_(self.model[8].weight,0,0.001)
    def forward(self, input):
        output = self.model(input)
        return output

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
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(H, 1, bias=True),
            torch.nn.Sigmoid()
        )
    def forward(self, input):
        output = self.model(input)
        return output


def train(model, total_S, total_labels, validation_S, validation_labels,
          batch_size=100, lr=0.0002, epochs=1000, load_epoch=0, save_per=10, momentum = 0.99):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99)    
    criterion = torch.nn.BCELoss().cuda()
    loss_records = []
    validation_records = np.zeros(epochs)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=2-1.0000002)
    for epoch in trange(load_epoch, epochs+load_epoch):
        # for g in optimizer.param_groups:
        #     g['momentum'] = 0.9+min(epoch,200)/200*0.09
        #     g['lr'] = max(lr*((1.0/1.0000002)**(epoch*len(total_S))), 1e-6)
        # print(g['momentum'], g['lr'])
        # print('epoch', epoch)
        order = np.random.permutation(len(total_S))
        for i in order:
            optimizer.zero_grad()
            S = total_S[i]
            labels = total_labels[i]
            outputs = model(S)
            #print(outputs)
            loss = criterion(outputs, labels)
            # for param in model.parameters():
            #     loss += 0.00001 * (param ** 2).sum()
            loss.backward()
            optimizer.step()
            #scheduler.step()

        validation_records[epoch] = criterion(model(validation_S), validation_labels).detach().cpu().numpy()
        if early_stopping(validation_records, epoch):
            path = './checkpoint%d/'%n+str(epoch)+'/'
            try:
                os.makedirs(path) 
            except:
                pass
            torch.save(model.state_dict(), path+'model.pt')
            break

        # if epoch % save_per == 0:
        #     path = './checkpoint%d/'%n+str(epoch)+'/'
        #     try:
        #         os.makedirs(path) 
        #     except:
        #         pass
        #     torch.save(model.state_dict(), path+'model.pt')
        #     #clear_output(True)
        #     loss_records.append(loss.detach().cpu().numpy())
        #     validation_records.append(criterion(model(validation_S), validation_labels).detach().cpu().numpy() )
        #     print("epoch: ", epoch)
        #     print("loss: ", loss)
        #     print("validation loss: ", validation_records[-1])
        #     #print("accuracy on X: ", inference(model, len(X), X))

        #     plt.plot(save_per*np.arange(len(loss_records)),loss_records)
        #     plt.plot(save_per*np.arange(len(validation_records)),validation_records)
        #     plt.show()
        #     plt.savefig('./checkpoint%d/'%n+'loss.png')

        #     #calculate_pval(model, dataset_P[n:2*n], dataset_Q[n:2*n], N=20)
            
    #raise Exception("Done training")
    return model

if __name__ == "__main__":
    try:
        title = sys.argv[1]
    except:
        title='untitled'
    device = torch.device("cuda:0")
    dtype = torch.float32

    ##### Data #####
    dataset = np.load('../HIGGS.npy')
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28), 0
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28), 1
    #dataset_P = np.zeros((100000,28))
    #dataset_Q = np.ones((100000,28))
    del dataset
    # 改这个
    # 1300000是paper里的
    # 1300001是我的，把lr=0.05改小成2e-3, batch从100变成2048
    # 130002: 300, 6, Tanh
    # 1299999t：300，6，ReLU
    #n = 50000 #####
    test_epoch = 1300002 #####

    if title == 'test':
        print('-------------------test--------------------')
        path = './checkpoint%d/'%n+str(test_epoch)+'/'
        model = DN().to(device)
        model.load_state_dict(torch.load(path+'model.pt'))
        pval = calculate_pval(model, dataset_P[n:2*n], dataset_Q[n:2*n], N=20)
        #pval = calculate_pval(model, dataset_P[0:n], dataset_Q[0:n], N=20)

    elif title == 'train':
        print('-------------------train--------------------')
        for n in [1300100, 1000100, 700100]:
            X, Y = dataset_P[:n], dataset_Q[:n]
            batch_size = 100
            batches = n//batch_size
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
            model = DN().to(device)
            #model.load_state_dict(torch.load('./Scheffe/checkpoint%d/'%n+str(130)+'/'+'model.pt'))
            model = train(model, total_S, total_labels, validation_S, validation_labels,
                        batch_size=batch_size, lr=2e-4, epochs=1001, load_epoch=0, save_per=10, momentum=0.99)

            