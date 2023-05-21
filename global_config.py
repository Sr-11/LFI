
import os
import inspect
current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe())))

method_configs = {
    'Mix': True,
    'Fea_Gau': True,
    'Gaussian': True,
    'Scheffe': True,
    'LBI': True,
    'UME': True,
    'RFM': True
}

expr_configs = {
    'assets_path': current_dir+'/assets',
}

resource_configs = {
    'Higgs_path': current_dir+'/datasets/HIGGS.npy',
}

train_param_configs = {
    'gpu': '7',
    'n_tr_list': [1300000, 1000000, 700000, 400000, 200000, 50000],
    'repeat': range(10),
    'patience': 10,
    'save_every': 1,
    'learning_rate': 0.01,
    'momentum': 0.9,
    'batch_size': 4096,
    'N_epoch': 501,
}

test_param_configs = {
    'gpu': '6',
    'n_tr_list': [1300000, 1000000, 700000, 400000, 200000, 50000],
    'num_models': range(10), # same as train.repeat
    'num_repeat': range(10),
    'n_cal': 32768,
    'n_ev': lambda n_tr: n_tr,
    'pi': 1/11,
    'm': 1100,
}




