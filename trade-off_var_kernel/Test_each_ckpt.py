import numpy as np
import torch
import sys
from tqdm import tqdm, trange
import os
from IPython.display import clear_output
sys.path.append(os.getcwd()+'/../')
import lfi
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
dtype = torch.float32
torch.manual_seed(42)
np.random.seed(42)
torch.set_grad_enabled(False)
import time

def main(gpu, n, overwrite):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda:0")
    dataset = np.load('../HIGGS.npy')   
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)
    dataset_P = torch.from_numpy(dataset_P).to(device, dtype)
    dataset_Q = torch.from_numpy(dataset_Q).to(device, dtype)
    del dataset

    sys.path.append('../methods/Res_Net')
    from model import Model
    
    if n==0:
        ns_total = np.array([1600000, 1300000, 1000000, 700000, 400000, 200000, 100000, 50000, 30000, 20000, 10000, 6000, 4500, 3000, 2000, 1000])
    else:
        ns_total = np.array([n])
    ms_total = 10**np.linspace(2, 7, 26)
    ms_total = ms_total.astype(int)
    repeats = 10
    pi = 0.1

    n_list_with_id = []
    for model_id in range(30):
        for n_tr in ns_total+model_id:
            n_list_with_id.append(n_tr)

    for n_tr in n_list_with_id:
        print('----- n_tr = %d -----'%n_tr)

        def callback():
            timing_mark = time.time()
            np.save('../methods/Res_Net/checkpoints/n_tr=%d/timing_mark.npy'%n_tr, timing_mark)

        flag_timing_mark_exist = os.path.exists('../methods/Res_Net/checkpoints/n_tr=%d/timing_mark.npy'%n_tr)
        if flag_timing_mark_exist: 
            flag_other_is_computing = (time.time()-np.load('../methods/Res_Net/checkpoints/n_tr=%d/timing_mark.npy'%n_tr) < 60)
        else:
            flag_other_is_computing = False
        flag_next_ckpt = os.path.exists('../methods/Res_Net/checkpoints/n_tr=%d/kernel.pt'%(n_tr+1))
        flag_computed = os.path.exists('../methods/Res_Net/checkpoints/n_tr=%d/type_1_error.npy'%n_tr)

        if flag_timing_mark_exist: print('Have timing mark')
        else: print('No timing mark')
        if flag_other_is_computing: print('Other process in this')
        else: print('No other process in this')
        if flag_next_ckpt: print('Next ckpt (n+1) exist')
        else: print('Next ckpt (n+1) not exist')
        if flag_computed: print('Error computed')
        else: print('Error not computed')
                                
        if (overwrite and flag_other_is_computing==False) or (flag_next_ckpt==True and flag_computed==False and flag_other_is_computing==False):
            callback()
            n_te = n_tr
            n_ev = 20000
            batch_size_for_score_X_test_Y_test_X_eval = 10000
            # if n_tr > 10000: 
            #     n_te = 10000
            kernel = torch.load('../methods/Res_Net/checkpoints/n_tr=%d/kernel.pt'%n_tr)
            print('Start Compute')
            plot_hist_path = '../methods/Res_Net/checkpoints/n_tr=%d/plot_hist.png'%n_tr

            type_1_error, type_2_error = lfi.test.simulate_error(dataset_P, dataset_Q,
                                                            n_tr, n_ev, n_te, 
                                                            kernel, repeats,
                                                            pi, ms_total,
                                                            batch_size_for_score_X_test_Y_test_X_eval,
                                                            plot_hist_path=plot_hist_path,
                                                            callback=callback)
            np.save('../methods/Res_Net/checkpoints/n_tr=%d/type_1_error.npy'%n_tr, type_1_error)
            np.save('../methods/Res_Net/checkpoints/n_tr=%d/type_2_error.npy'%n_tr, type_2_error)
            print('Finish Compute')
            print('type_1_error:')
            print(np.mean(type_1_error, axis=1))
            print('type_2_error:')
            print(np.mean(type_2_error, axis=1))
        else:
            print('Skip')

if __name__ == '__main__':
    gpu = sys.argv[1]
    try:
        n = int(sys.argv[2])
    except:
        n = 0
    try:
        overwrite = sys.argv[3]
    except:
        overwrite = False
    main(gpu, n, overwrite)
