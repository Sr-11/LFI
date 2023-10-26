
import sys
sys.path.append(sys.path[0]+"/../..") 
import inspect
import importlib.util
import os

current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))) 
global_config_path = current_dir+'/../../global_config.py'
global_config_spec = importlib.util.spec_from_file_location('config',global_config_path)
global_config = importlib.util.module_from_spec(global_config_spec)
global_config_spec.loader.exec_module(global_config)
from global_config import *

resource_configs = {
    'Higgs_path': global_config.resource_configs['Higgs_path'],
}

train_param_configs = {
    'gpu': '4',
    'n_tr_list': [1300000, 1000000, 700000, 400000, 200000, 50000],
    'repeats': range(10),
    'batch_size': 20000,
    'N_epoch': 501,
    'patience': 10,
    'median_heuristic': True,
}

test_param_configs = {
    'gpu': '4',
    'n_tr_list': [1300000, 1000000, 700000, 400000, 200000, 50000],
    'num_models': range(10),
    'num_repeat': range(10),
    'n_cal': 10000,
    'n_ev': 10000,
    'test_hard': False,
    'test_soft': True,
}




