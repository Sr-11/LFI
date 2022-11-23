#Implements Scheffes test by first building a classifier between X and Y, then classifies Y.
import numpy as np
import torch
import sys
from sklearn.utils import check_random_state
from matplotlib import pyplot as plt
from tqdm import trange
import pickle
from LFI import *
from Data_gen import *

class Classifier(torch.nn.Module):
    """Latent space for both domains."""
    """ Dense Net with w=50, d=4, ~relu, in=2, out=50 """
    def __init__(self, x_in, H):
        """Init latent features."""
        super(Classifier, self).__init__()
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(H, 2, bias=True),
            torch.nn.Softmax()
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant



def train(model, X, Y, criterion, batch_size=64, lr=0.001, epochs=1000):
    """Label the items first
        items in X have label 1, items in Y have label 0
        then train the model with sgd
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        print("epoch: ", epoch)
        for i in range(0, len(X), batch_size):
            optimizer.zero_grad()
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            output_X = model(X_batch)
            output_Y = model(Y_batch)
            loss_X = criterion(output_X, torch.ones(len(X_batch), 1).to(device))
            loss_Y = criterion(output_Y, torch.zeros(len(Y_batch), 1).to(device))
            loss = loss_X + loss_Y
            loss.backward()
            optimizer.step()
    

def inference(model,size_m, parameters, trial=1000):
    '''
    for t in range(trial):
        Z, _ = gen_data(size_m, parameters)
        Z = Z.to(device)
        class=torch.sum(model(Z))
        success+=(class.item()==0)
    '''
    pass


if __name__ == "__main__":
    device = torch.device("cuda")
    try:
        title=sys.argv[1]
    except:
        title='untitled_run'
    x_in=2
    H=50
    model=Classifier(x_in, H)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    n=100
    sigma_mx_2=np.ones((3,3))
    X, Y=sample_blobs_Q(n, sigma_mx_2)
    model=train(model, X, Y, criterion)
    