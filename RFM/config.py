
# def checkpoints_path(cur_path, n_tr):
#     return cur_path+'/checkpoints/checkpoints n_tr=%d/'%n_tr
import sys
sys.path.append(sys.path[0]+"/..") 
import global_config


expr_configs = {
    'checkpoints_path': sys.path[0]+'/checkpoints',
    'pval_mat_path': sys.path[0]+'/pval_data'
}

resource_configs = {
    'Higgs_path': sys.path[0]+'/../HIGGS.npy',
}

train_param_configs = {
    'gpu_id': '7',
    # 'n_tr_list': [1300000, 1000000, 700000, 400000, 200000, 50000],
    'n_tr_list': global_config.train_param_configs['n_tr_list'],
    'repeats': global_config.train_param_configs['repeats'],
    'batch_size': 10020,
    'N_epoch': 501,
}

test_param_configs = {
    'gpu_id': '6',
    # 'n_tr_list': [1300000, 1000000, 700000, 400000, 200000, 50000],
    'n_tr_list': global_config.test_param_configs['n_tr_list'],
    'num_models': global_config.test_param_configs['num_models'], # same as train.repeats
    'num_repeats': global_config.test_param_configs['num_repeats'],
    'n_ev': 10000,
    'n_te': 10000,
    'test_hard': False,
    'test_soft': True,
}




