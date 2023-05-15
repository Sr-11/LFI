
# def checkpoints_path(cur_path, n_tr):
#     return cur_path+'/checkpoints/checkpoints n_tr=%d/'%n_tr
import sys
sys.path.append(sys.path[0]+"/../..") 
import global_config

expr_configs = {
    'checkpoints_path': sys.path[0]+'/checkpoints',
    'pval_mat_path': sys.path[0]+'/pval_data'
}

resource_configs = {
    'Higgs_path': global_config.resource_configs['Higgs_path'],
}

train_param_configs = {
    'gpu_id': '0',
    'n_tr_list': global_config.train_param_configs['n_tr_list'],
    'repeats': global_config.train_param_configs['repeats'],
    'J_tr': 2048,
    'batch_size': 2048,
    'N_epoch': 501,
    'learning_rate': 0.01,
    'momentum': 0.9,
}

test_param_configs = {
    'gpu_id': '2',
    'n_tr_list': global_config.test_param_configs['n_tr_list'],
    'num_models': global_config.test_param_configs['num_models'], # same as train.repeats
    'num_repeats': global_config.test_param_configs['num_repeats'],
    'n_ev': 10000,
    'n_te': 10000,
    'test_hard': True,
    'test_soft': True,
}




