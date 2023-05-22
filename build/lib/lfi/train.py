import numpy as np
import torch
import sys, os, gc
import importlib.util
from matplotlib import pyplot as plt
# import cProfile
from tqdm import tqdm
from IPython.display import clear_output
import json
import inspect
import argparse
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))))
from lfi.utils import *

device = torch.device("cuda:0")

# define the pre-processing function
def pre_process(dataset_P, dataset_Q, n, batch_size):
    def split_number(n, batches):
        quotient = n//batches
        remainder = n%batches
        split_size_list = [quotient] * batches
        for i in range(remainder):  
            split_size_list[i] += 1
        return split_size_list

    batches = (n-1)//batch_size + 1
    split_size_list = split_number(n, batches)
    cumulative_sum = np.cumsum([0]+split_size_list)
    X = dataset_P[0:n]
    Y = dataset_Q[0:n]
    total_S = [(X[cumulative_sum[i]: cumulative_sum[i+1]], 
                Y[cumulative_sum[i]: cumulative_sum[i+1]])
                for i in range(len(split_size_list))]
    total_S = [MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in total_S]
    return total_S

# define the early stopping criterion
def early_stopping(validation_losses, patience):
    return len(validation_losses) - np.argmin(validation_losses) >= patience

# find epoch-patience checkpoint
def maintain_checkpoints(dir, patience, kernel, kernel_histories):
    torch.save(kernel, dir+'/kernel_-0.pt')
    kernel_copy = torch.load(dir+'/kernel_-0.pt')
    kernel_histories.append(kernel_copy)
    assert len(kernel_histories) == patience+1, 'kernel_histories length = %d, patience = %d'%(len(kernel_histories), patience)
    for i in range(0, patience+1):
        if kernel_histories[patience-i] != None:
            torch.save(kernel_histories[patience-i], dir+'/kernel_-%d.pt'%(i))
    return kernel_histories.pop(0)
    
# first level of training, one epoch
def one_epoch(epoch, kernel, train_loader, optimizer, train_loss_records):
    # torch.distributed.get_rank()
    if 'one_epoch' not in dir(kernel):
        train_loader_iter = iter(train_loader)
        for XY_train in tqdm(train_loader_iter):
            optimizer.zero_grad()
            obj = kernel.compute_loss(XY_train[0,:,:])
            obj.backward()
            optimizer.step()   
            train_loss_records.append(obj.item())
        kernel.epoch = epoch
    else:
        kernel.one_epoch(epoch, train_loader, optimizer, train_loss_records)

# second level of trainig, train each n_train
def train(n_tr, train_loader, S_validate, 
          kernel, optimizer, N_epoch, 
          save_every, ckpt_dir, patience, 
          **kwargs):  
    J_validation_records = []
    train_loss_records = []
    kernel_histories = [None]*patience
    kernel.epoch = 0
    maintain_checkpoints(ckpt_dir, patience, kernel, kernel_histories)
    # start training
    for epoch in range(N_epoch):
        # run one epoch
        print('----- n_tr =', n_tr, ', epoch =', epoch, '-----')
        one_epoch(epoch, kernel, train_loader, optimizer, train_loss_records)
        # validation
        with torch.no_grad():
            J_validation_records.append( kernel.compute_loss(S_validate).item() )
        print('validation =', J_validation_records[-1])
        # early stopping
        stop_flag = early_stopping(J_validation_records, patience)
        last_patience_kernel = maintain_checkpoints(ckpt_dir, patience, kernel, kernel_histories)
        # print
        if epoch%save_every == 0 or stop_flag == True:
            plt.plot(J_validation_records); plt.xlabel('epoch'); plt.ylabel('validation loss')
            plt.savefig(ckpt_dir+'/validation_loss_epoch.png'); plt.clf()
            plt.plot(train_loss_records); plt.xlabel('iteration'); plt.ylabel('training loss')
            plt.savefig(ckpt_dir+'/train_loss_iter.png'); plt.clf()
            plt.close(); plt.close('all')
        # early stopping
        if stop_flag:
            torch.save(last_patience_kernel, ckpt_dir+'/kernel.pt')
            return last_patience_kernel, J_validation_records
    torch.save(kernel, ckpt_dir+'/kernel.pt')
    return kernel, J_validation_records

# third level of trainig, run through all n_tr
def main(config_dir, **kwargs): 
    config_path = os.path.join(config_dir, 'config.py')
    model_path = os.path.join(config_dir, 'model.py')
    checkpoints_path = os.path.join(config_dir, 'checkpoints')
    # import config
    config_spec = importlib.util.spec_from_file_location('config', config_path)
    config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config)
    # import neural model
    sys.path.append(os.path.abspath(os.path.dirname(model_path)))
    from model import Model
    # import data
    dataset = np.load(config.resource_configs['Higgs_path'])
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5170877, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5829122, 28) 
    # load parameters
    os.environ["CUDA_VISIBLE_DEVICES"]=config.train_param_configs['gpu']
    n_tr_list = config.train_param_configs['n_tr_list']
    repeat = config.train_param_configs['repeat']
    batch_size = config.train_param_configs['batch_size']
    N_epoch = config.train_param_configs['N_epoch']
    learning_rate = config.train_param_configs['learning_rate']
    momentum = config.train_param_configs['momentum']
    save_every = config.train_param_configs['save_every']
    patience = config.train_param_configs['patience']
    # load parameters from kwargs
    if 'n_tr_list' in kwargs:
        n_tr_list = config.train_param_configs['n_tr_list'] = json.loads(kwargs['n_tr_list'])
    if 'repeat' in kwargs:
        repeat = config.train_param_configs['repeat'] = json.loads(kwargs['repeat'])
    if 'patience' in kwargs:
        patience = config.train_param_configs['patience'] = int(kwargs['patience'])
    if 'batch_size' in kwargs:
        batch_size = config.train_param_configs['batch_size'] = kwargs['batch_size']
    if 'gpu' in kwargs:
        os.environ["CUDA_VISIBLE_DEVICES"]= kwargs['gpu']
    if 'checkpoints_path' in kwargs:
        checkpoints_path = kwargs['checkpoints_path']
    # load model init parameters
    Model_init_kwargs = {}
    if 'median_heuristic' in kwargs:
        median_heuristic = config.train_param_configs['median_heuristic'] = (kwargs['median_heuristic']=='True')
        Model_init_kwargs['median_heuristic'] = median_heuristic
    if 'median_heuristic' in config.train_param_configs.keys():
        median_heuristic = config.train_param_configs['median_heuristic']
        Model_init_kwargs['median_heuristic'] = median_heuristic
    # if Model_init_kwargs['median_heuristic'] == True:


    # train
    n_raise_error_list = []
    for n in n_tr_list:
        # train loader
        if n>1e4:
            validate_size = int(10*np.sqrt(n))
        if n<1e4:
            validate_size = int(0.1*n)
        
        total_S = pre_process(dataset_P, dataset_Q, n-validate_size, batch_size)
        train_loader = torch.utils.data.DataLoader(total_S, batch_size=1, shuffle=True)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_loader)
        for r in repeat:
            n_raise_error_list.append(n)
            for _ in range(2):
                print('\n----------- TRAIN n_tr = %d, repeated = %d ----------'%(n,r))
                # validation
                S_validate = np.concatenate((dataset_P[n+np.random.choice(dataset_P.shape[0]-n, validate_size, replace=False)],  dataset_Q[n+np.random.choice(dataset_Q.shape[0]-n, validate_size, replace=False)]), axis=0)
                S_validate = MatConvert(S_validate, device, dtype)
                # edit Model_init_kwargs
                Model_init_kwargs['XY_heu'] = total_S[0]
                if 'J' in config.train_param_configs.keys(): # only for UME
                    J = config.train_param_configs['J']
                    Model_init_kwargs['J'] = J
                    V = np.concatenate((dataset_P[np.random.choice(n, J//2, replace=False)],  dataset_Q[np.random.choice(n, J//2, replace=False)]), axis=0)
                    V = MatConvert(V, device, dtype)
                    Model_init_kwargs['V'] = V
                # model
                kernel = Model(device=device, **Model_init_kwargs)
                # kernel = torch.nn.parallel.DistributedDataParallel(kernel)
                optimizer = torch.optim.SGD(kernel.params, lr=learning_rate, momentum=momentum)
                # checkpoint path
                ckpt_dir = checkpoints_path + '/n_tr=%d#%d'%(n,r)
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                # train
                kernel, Js = train(n, train_loader, S_validate, 
                                kernel, optimizer, 
                                N_epoch, save_every, ckpt_dir, patience,
                                dataset_P=dataset_P, dataset_Q=dataset_Q)
                print('best epoch:', kernel.epoch)
                # see if training failed
                if min(Js)<max(Js)-0.01:
                    n_raise_error_list.pop()
                    break
                else:
                    print('------------------------------------------')
                    print('----- WARNING: training might failed -----')
                    print('---- Check the loss-epoch plot in ckpt ----')
                    print('------------------------------------------')
    
    if len(n_raise_error_list)>0:
        print('Training might failed at n = ', n_raise_error_list)
        
if __name__ == "__main__":
    current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))) 
    method = sys.argv[1]
    config_dir = os.path.join(current_dir, '..', 'methods', method)
    if method != 'RFM':
        main(config_dir, **dict([arg.split('=') for arg in sys.argv[2:]]))
    else:
        os.system('python %s %s'%(os.path.join(config_dir, 'RFM_train.py'), ' '.join(sys.argv[2:])))
