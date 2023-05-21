
from lfi.utils import *
import torch
import numpy as np

class Classifier(torch.nn.Module):
    def __init__(self, H=300):
        super(Classifier, self).__init__()
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
        )
    def forward(self, input):
        output = self.model(input)
        return output

class Model(torch.nn.Module):
    def __init__(self, device, **kwargs):
        super(Model, self).__init__()
        self.device = device
        self.model = Classifier().to(device)
        self.params = list(self.model.parameters())
        self.criterion = torch.nn.BCEWithLogitsLoss().to(device)
    def compute_loss(self, XY_tr, require_grad=True, **kwargs):
        preds = self.model(XY_tr)
        batch_size = XY_tr.shape[0]//2
        labels = torch.cat((torch.zeros((batch_size,1), dtype=dtype), torch.ones((batch_size,1), dtype=dtype)) ).to(self.device)
        loss = self.criterion(preds, labels)
        return loss
    def compute_scores(self, X_ev, Y_ev, Z_input, batch_size=1024):
        Z_input_splited = torch.split(Z_input, batch_size)
        phi_Z = torch.zeros(Z_input.shape[0]).to(self.device)
        for i, Z_input_batch in enumerate(Z_input_splited):
            phi_Z[i*batch_size: i*batch_size+Z_input_batch.shape[0]] = self.model(Z_input_batch).squeeze()
        return phi_Z
    def compute_gamma(self, X_ev, Y_ev, pi):
        return pi/2



