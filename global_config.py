
# def checkpoints_path(cur_path, n_tr):
#     return cur_path+'/checkpoints/checkpoints n_tr=%d/'%n_tr
import sys, os
current_dir = os.path.dirname(os.path.realpath(__file__))

method_configs = {
    'Res_Net': True,
    'Mix': False,
    'Fea_Gau': True,
    'Gaussian': True,

    'Scheffe': True,
    'LBI': True,

    'UME': True,
    'RFM': True
}

expr_configs = {
    'checkpoints_path': current_dir+'/checkpoints',
    'pval_mat_path': current_dir+'/pval_data'
}

resource_configs = {
    'Higgs_path': current_dir+'/HIGGS.npy',
}

train_param_configs = {
    'repeats': 5,
    'n_tr_list': [1300000, 1000000, 700000, 400000, 200000, 50000],

    'gpu_id': '7',
    'batch_size': 1024,
    'N_epoch': 501,
    'print_every': 10,
}

test_param_configs = {
    'n_tr_list': [1300000, 1000000, 700000, 400000, 200000, 50000],
    'num_models': 5, # same as train.repeats
    'num_repeats': 5,

    'gpu_id': '6',
    'n_ev': 10000,
    'n_te': 10000,
    'test_hard': True,
    'test_soft': True,
}




