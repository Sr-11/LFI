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



def train(model, X, Y, criterion):
    pass


def inference(model, parameters, trial=1000):
    '''
    for t in range(trial):
        Z, _ = gen_data(m, parameters)
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
    train(model, X, Y, )