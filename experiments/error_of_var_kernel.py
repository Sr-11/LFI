import numpy as np
import torch
import sys
from tqdm import tqdm, trange
import os
from IPython.display import clear_output
import inspect
cur_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(cur_dir+'/..')
sys.path.append(cur_dir+'/../..')
print(os.getcwd())
import lfi
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
device = torch.device("cuda:0")
dtype = torch.float32
torch.manual_seed(42)
np.random.seed(42)
torch.set_grad_enabled(False)
import time

def main(gpu, n='default', overwrite=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda:0")
    dataset = np.load(cur_dir+'/../datasets/HIGGS.npy')   
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)
    dataset_P = torch.from_numpy(dataset_P).to(device, dtype)
    dataset_Q = torch.from_numpy(dataset_Q).to(device, dtype)
    del dataset

    sys.path.append(cur_dir+'/../methods/Mix')
    from model import Model
    if n=='default':
        ns_total = np.array([1600000, 1300000, 1000000, 700000, 400000, 200000, 100000, 50000, 30000, 20000, 10000, 6000, 4500, 3000, 2000, 1000, 500, 200, 100])
    else:
        ns_total = np.array([n])
    ms_total = 10**np.linspace(2, 7, 26)
    ms_total = ms_total.astype(int)
    repeats = range(10)
    pi = 0.1

    for n_tr in ns_total:
        for r in repeats:
            print('----- n_tr = %d -----'%n_tr)

            if not os.path.exists(cur_dir+'/../methods/Mix/checkpoints/n_tr=%d#%d'%(n_tr,r)):
                print()
                print('YOU DID NOT FINISH THE TRAINING!')
                print()
            def callback():
                timing_mark = time.time()
                np.save(cur_dir+'/../methods/Mix/checkpoints/n_tr=%d#%d/timing_mark.npy'%(n_tr,r), timing_mark)

            flag_timing_mark_exist = os.path.exists(cur_dir+'/../methods/Mix/checkpoints/n_tr=%d#%d/timing_mark.npy'%(n_tr,r))
            if flag_timing_mark_exist: 
                flag_other_is_computing = (time.time()-np.load(cur_dir+'/../methods/Mix/checkpoints/n_tr=%d#%d/timing_mark.npy'%(n_tr,r)) < 60)
            else:
                flag_other_is_computing = False
            flag_next_ckpt = os.path.exists(cur_dir+'/../methods/Mix/checkpoints/n_tr=%d#%d/kernel.pt'%(n_tr,r+1))
            flag_computed = os.path.exists(cur_dir+'/../methods/Mix/checkpointsn_tr=%d#%d/type_1_error.npy'%(n_tr,r))

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
                kernel = torch.load(cur_dir+'/../methods/Mix/checkpoints/n_tr=%d#%d/kernel.pt'%(n_tr,r))
                print('Start Compute')
                plot_hist_path = cur_dir+'/../methods/Mix/checkpoints/n_tr=%d#%d/plot_hist.png'%(n_tr,r)

                type_1_error, type_2_error = lfi.test.simulate_error(dataset_P, dataset_Q,
                                                                n_tr, n_ev, n_te, 
                                                                kernel, [r],
                                                                pi, ms_total,
                                                                batch_size_for_score_X_test_Y_test_X_eval,
                                                                plot_hist_path=plot_hist_path,
                                                                callback=callback)
                np.save(cur_dir+'/../methods/Mix/checkpoints/n_tr=%d#%r/type_1_error.npy'%(n_tr,r), type_1_error)
                np.save(cur_dir+'/../methods/Mix/checkpoints/n_tr=%d#%r/type_2_error.npy'%(n_tr,r), type_2_error)
                print('Finish Compute')
                print('type_1_error:')
                print(np.mean(type_1_error, axis=1))
                print('type_2_error:')
                print(np.mean(type_2_error, axis=1))
            else:
                print('Skip')

if __name__ == '__main__':
    try:
        gpu = sys.argv[1]
    except:
        gpu = '0'
    # try:
    #     n = int(sys.argv[2])
    # except:
    #     n = 'defualt'
    # try:
    #     overwrite = sys.argv[3]
    # except:
    #     overwrite = False
    main(gpu)
