
# def checkpoints_path(cur_path, n_tr):
#     return cur_path+'/checkpoints/checkpoints n_tr=%d/'%n_tr
import sys, os
import importlib.util

current_dir = os.path.dirname(os.path.realpath(__file__))
global_config_path = current_dir+'/../../global_config.py'
global_config_spec = importlib.util.spec_from_file_location('config',global_config_path)
global_config = importlib.util.module_from_spec(global_config_spec)
global_config_spec.loader.exec_module(global_config)

model_configs = {
    'model_path': current_dir+'/model.py',
}

expr_configs = {
    'checkpoints_path': current_dir+'/checkpoints',
    'pval_mat_path': current_dir+'/pval_data',
    'plot_path': current_dir+'/plots',
}

resource_configs = {
    'Higgs_path': global_config.resource_configs['Higgs_path'],
}

train_param_configs = {
    'gpu_id': '1',
    'n_tr_list': global_config.train_param_configs['n_tr_list'],
    'repeats': global_config.train_param_configs['repeats'],
    'batch_size': 1024,
    'N_epoch': 501,
    'validation_size': 2048,
    'learning_rate': 0.01,
    'momentum': 0.9,
    'save_every': 5,
    'patience': 10,
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
    'force_thres': None,

}




