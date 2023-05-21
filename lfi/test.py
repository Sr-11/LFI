import numpy as np
import torch
import sys, os, gc
from tqdm import tqdm, trange
from IPython.display import clear_output
import importlib.util
import time
import inspect
import json
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))))
from lfi.utils import *

device = torch.device("cuda:0")

def simulate_p_value(dataset_P, dataset_Q, 
                    n_tr, n_cal, n_ev,
                    kernel, repeats,
                    pi, m,
                    test_soft, test_hard, force_thres,
                    plot_hist_path,
                    print_cal_opt_dist=False,
                    **kwargs):
    """simulate p-value by 'repeats' times"""
    with torch.no_grad():
        p_soft_list = np.zeros(len(repeats)) # soft: use T=\sum f(Z)
        p_hard_list = np.zeros(len(repeats)) # hard: use T=\sum 1{f(Z)>t}
        p_force_thres_list = np.zeros(len(repeats))
        # run many times
        for r in tqdm(repeats, desc='progress of repeating %d times when n_tr=%d'%(len(repeats),n_tr)):
            X_ev = dataset_P[ np.random.choice(n_tr, n_ev, replace=False) ]
            Y_ev = dataset_Q[ np.random.choice(n_tr, n_ev, replace=False) ]
            X_cal = dataset_P[ n_tr + np.random.choice(dataset_P.shape[0]-n_tr, n_cal, replace=False) ]
            Y_cal = dataset_Q[ n_tr + np.random.choice(dataset_Q.shape[0]-n_tr, n_cal, replace=False) ]
            # Compute the test statistic on the test data
            X_scores = kernel.compute_scores(X_ev, Y_ev, X_cal) # (n_cal,)
            Y_scores = kernel.compute_scores(X_ev, Y_ev, Y_cal) # (n_cal,)
            # compute p-value
            if test_soft:
                p_soft_list[r] = get_pval_from_evaluated_scores(X_scores, Y_scores, pi, m, thres=None, verbose = False)
                print("pval_original : %.2f"%p_soft_list[r])
            if test_hard:
                X_opt = dataset_P[ n_tr+np.random.choice(dataset_P.shape[0]-n_tr, n_cal, replace=False) ]
                Y_opt = dataset_Q[ n_tr+np.random.choice(dataset_Q.shape[0]-n_tr, n_cal, replace=False) ]
                X_opt_scores = kernel.compute_scores(X_ev, Y_ev, X_opt)
                Y_opt_scores = kernel.compute_scores(X_ev, Y_ev, Y_opt) 
                t_opt, thres_opt_list, pval_opt_list = get_thres_from_evaluated_scores(X_opt_scores, Y_opt_scores, pi, m, 
                                                              os.path.join(plot_hist_path,'ROC.png'))
                p_hard_list[r] = get_pval_from_evaluated_scores(X_scores, Y_scores,  pi, m, thres=t_opt, verbose = False)
                print("pval_(t_opt=%.3f) : %.2f"%(t_opt, p_hard_list[r]))
                if print_cal_opt_dist:
                    print('start cal dist')
                    t_cal, thres_cal_list, pval_cal_list = get_thres_from_evaluated_scores(X_scores, Y_scores, pi, m)
                    plot_pval_thres(os.path.join(plot_hist_path,'pval-thres.png'), [t_opt, thres_opt_list, pval_opt_list], [t_cal, thres_cal_list, pval_cal_list])
            else:
                t_opt = None
            if force_thres != None:
                p_force_thres_list[r] = get_pval_from_evaluated_scores(X_scores, Y_scores, thres=force_thres, verbose = False)
                print("pval_(t=%.2f) : %.2f"%(force_thres, p_force_thres_list[r]))

            # plot histogram 
            if plot_hist_path != None:
                title = 'n_tr=%d, n_cal=%d, n_ev=%d'%(n_tr, n_cal, n_ev)
                plot_hist(X_scores, Y_scores, os.path.join(plot_hist_path,'hist.png'), title=title, pi=pi, thres=t_opt, verbose=True)

            del X_ev, Y_ev, X_cal, Y_cal, X_scores, Y_scores
            gc.collect(); torch.cuda.empty_cache()
        return p_soft_list, p_hard_list, p_force_thres_list

def main_pval(config_dir, **kwargs):
    with torch.no_grad():
        config_path = os.path.join(config_dir, 'config.py')
        model_dir = config_dir
        chekpoints_path = os.path.join(config_dir, 'checkpoints')
        # import config
        config_spec = importlib.util.spec_from_file_location('config', config_path)
        config = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config)
        # import neural model
        sys.path.append(model_dir)
        from model import Model
        # load params
        n_tr_list = config.test_param_configs['n_tr_list']
        n_cal = config.test_param_configs['n_cal']
        n_ev = config.test_param_configs['n_ev']
        num_models = config.test_param_configs['num_models']
        num_repeat = config.test_param_configs['num_repeat']
        test_soft = config.test_param_configs['test_soft']
        test_hard = config.test_param_configs['test_hard']
        force_thres = config.test_param_configs['force_thres']
        pi = config.test_param_configs['pi']
        m = config.test_param_configs['m']
        if 'n_tr_list' in kwargs.keys():
            n_tr_list = json.loads(kwargs['n_tr_list'])
        if 'gpu' in kwargs.keys():
            os.environ["CUDA_VISIBLE_DEVICES"]= kwargs['gpu']
        else:
            os.environ["CUDA_VISIBLE_DEVICES"]= config.test_param_configs['gpu'] 
        # load data
        dataset = np.load(config.resource_configs['Higgs_path'])
        dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
        dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)
        dataset_P = torch.from_numpy(dataset_P).to(dtype=dtype, device=device)
        dataset_Q = torch.from_numpy(dataset_Q).to(dtype=dtype, device=device)
        # run tests
        for n_tr in n_tr_list:
            for r in num_models:
                print("\n------------------- TEST n_tr = %d, repeated trained network = %d -------------------"%(n_tr,r))
                ckpt_dir = chekpoints_path+'/n_tr=%d#%d'%(n_tr,r)
                # ckpt_dir = chekpoints_path+'/n_tr=%d'%(n_tr+r)
                # ckpt_dir = chekpoints_path+'/n_tr=%d'%(n_tr)
                kernel = torch.load(ckpt_dir+'/kernel.pt')
                plot_hist_path = ckpt_dir
                p_soft_list, p_hard_list, p_force_thres_list = simulate_p_value(dataset_P, dataset_Q,
                                                                            n_tr, n_cal, n_ev(n_tr),
                                                                            kernel, num_repeat,
                                                                            pi, m, 
                                                                            test_soft, test_hard, force_thres,
                                                                            plot_hist_path,
                                                                            **kwargs)
                if test_soft:
                    np.save(ckpt_dir+'/pval_orig.npy', p_soft_list)
                    print("pval_orig_mean = ", np.mean(p_soft_list))
                    print("pval_orig_std = ", np.std(p_soft_list))
                if test_hard:
                    np.save(ckpt_dir+'/pval_t_opt.npy', p_hard_list)
                    print("pval_t_opt_mean = ", np.mean(p_hard_list))
                    print("pval_t_opt_std = ", np.std(p_hard_list))
                if force_thres != None:
                    np.save(ckpt_dir+'/pval_force_t.npy', p_force_thres_list)
                    print("p_force_thres_mean = ", np.mean(p_force_thres_list))
                    print("p_force_thres_std = ", np.std(p_force_thres_list))
            gc.collect(); torch.cuda.empty_cache()
            clear_output(wait=True)



def simulate_error(dataset_P, dataset_Q,
                   n_tr, n_cal, n_ev, 
                   kernel, repeats,
                   pi, m,
                   batch_size,
                   plot_hist_path=None,
                   callback=None,):
    """simulate test error by 'repeats' times"""
    with torch.no_grad():
        m_num = m.shape[0]
        type_1_error_list = np.zeros([m_num, repeats])
        type_2_error_list = np.zeros([m_num, repeats])
        # run many times
        for j in tqdm(range(repeats), desc='progress of repeating in n_tr=%d'%n_tr):
            idx = np.random.choice(dataset_P.shape[0]-n_tr, n_ev+n_cal, replace=False)+n_tr
            idy = np.random.choice(dataset_Q.shape[0]-n_tr, n_ev+n_cal, replace=False)+n_tr
            X_ev = dataset_P[ idx[:n_ev] ]
            X_cal = dataset_P[ idx[n_ev:] ]
            Y_ev = dataset_Q[ idy[:n_ev] ]
            Y_cal = dataset_Q[ idy[n_ev:] ]
            # Compute the test statistic on the test data
            X_scores = kernel.compute_scores(X_ev, Y_ev, X_cal, batch_size=batch_size) # shape=(n_cal,)
            Y_scores = kernel.compute_scores(X_ev, Y_ev, Y_cal, batch_size=batch_size) # shape=(n_cal,)
            gamma = kernel.compute_gamma(X_ev, Y_ev, pi)
            # gamma = (1-pi/2)*torch.mean(X_scores) + pi/2*torch.mean(Y_scores)
            type_1_error, type_2_error = get_error_from_evaluated_scores(X_scores, Y_scores, pi, gamma, m)
            type_1_error_list[:, j] = type_1_error
            type_2_error_list[:, j] = type_2_error
            # plot hist
            if plot_hist_path != None:
                title = 'n_cal=%d, n_ev=%d, n_tr=%d, repeat=%d'%(n_cal, n_ev, n_tr, j)
                plot_hist(X_scores, Y_scores, plot_hist_path, title, pi=pi, gamma=gamma, verbose=True)
            # delete to release memory
            del X_scores, Y_scores, gamma; gc.collect(); torch.cuda.empty_cache()
            # callback
            if callback != None:
                callback()
        return type_1_error_list, type_2_error_list

def main_error(config_path, n_list, m_list, method='Mix', pi=0.1, **kwargs):
    # load config
    current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))) 
    config_path = os.path.join(current_dir, '..', 'methods', method, 'config.py')
    config_spec = importlib.util.spec_from_file_location('config', config_path)
    config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config)
    # load params
    if 'gpu' in kwargs.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] = kwargs['gpu']; 
    if 'n_list' in kwargs.keys():
        n_list = kwargs['n']
    if 'overwrite' in kwargs.keys():
        overwrite = kwargs['overwrite']
    # load data
    dataset = np.load(config.resource_configs['Higgs_path'])
    dataset_P = dataset[dataset[:,0]==0][:, 1:] # background (5829122, 28)
    dataset_Q = dataset[dataset[:,0]==1][:, 1:] # signal     (5170877, 28)
    dataset_P = torch.from_numpy(dataset_P).to(device, dtype)
    dataset_Q = torch.from_numpy(dataset_Q).to(device, dtype)
    del dataset
    # load model
    config_spec = importlib.util.spec_from_file_location('config', config_path)
    config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config)
    sys.path.append(os.path.join('..', 'methods/', method))
    from model import Model
    # test
    with torch.no_grad():
        for n_tr in n_list:
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

                type_1_error, type_2_error = simulate_error(dataset_P, dataset_Q,
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

if __name__ == "__main__":
    current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))) 
    method = sys.argv[1]
    config_dir = os.path.join(current_dir, '..', 'methods', method)
    pval_or_error = sys.argv[2]

    if pval_or_error == 'pval':
        if method != 'RFM':
            main_pval(config_dir, **dict([arg.split('=') for arg in sys.argv[3:]]))
        else:
            os.system('python '+os.path.join(current_dir,'..','methods','RFM','RFM_test.py') +' '+ ' '.join(sys.argv[3:]))
    elif pval_or_error == 'error':
        main_error()