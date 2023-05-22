
# def checkpoints_path(cur_path, n_tr):
#     return cur_path+'/checkpoints/checkpoints n_tr=%d/'%n_tr
import sys, os
import importlib.util
import numpy as np
import inspect

current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))) 
global_config_path = current_dir+'/../../global_config.py'
global_config_spec = importlib.util.spec_from_file_location('config',global_config_path)
global_config = importlib.util.module_from_spec(global_config_spec)
global_config_spec.loader.exec_module(global_config)
from global_config import *


train_param_configs.update({
    'n_tr_list': np.array([1300000, 1600000, 1000000, 700000, 400000, 200000, 100000, 50000, 30000, 20000, 10000, 6000, 4500, 3000, 2000, 1000, 500, 200, 100]),
    'batch_size': 4096,
    'N_epoch': 501,
    'learning_rate': 0.01,
    'momentum': 0.9,
    'save_every': 1,
    'patience': 10,
    'median_heuristic': True,
})

test_param_configs.update({
    'pi': 1/11,
    'm': 1100,
    'test_hard': True,
    'test_soft': True,
    'force_thres': None,
})




