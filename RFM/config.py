
# def checkpoints_path(cur_path, n_tr):
#     return cur_path+'/checkpoints/checkpoints n_tr=%d/'%n_tr
import sys

expr_configs = {
    'checkpoints_path': sys.path[0]+'/checkpoints',
    'pval_mat_path': sys.path[0]+'/pval_data'
}

resource_configs = {
    'Higgs_path': '/math/home/eruisun/Datasets/HIGGS.npy',
}

train_param_configs = {
    'gpu_id': '7',
    'n_tr_list': [1300000, 1000000, 700000, 400000, 200000, 50000],
    'repeats': 1,
    'batch_size': 10000,
    'N_epoch': 501,
}

test_param_configs = {
    'gpu_id': '6',
    'n_tr_list': [1300000, 1000000, 700000, 400000, 200000, 50000],
    'num_models': 1, # same as train.repeats
    'num_repeats': 1,
    'n_ev': 10000,
    'n_te': 10000,
    'test_hard': True,
    'test_soft': True,
}




