
# def checkpoints_path(cur_path, n_tr):
#     return cur_path+'/checkpoints/checkpoints n_tr=%d/'%n_tr
expr_configs = {
    'checkpoints_path': '/checkpoints',
    'pval_mat_path': '/pval_data'
}

resource_configs = {
    'Higgs_path': './Datasets/HIGGS.npy',
}

train_param_configs = {
    'gpu_id': '0',
    'n_tr_list': [1300000, 1000000, 700000, 400000, 200000, 50000],
    'repeats': 10,
    'J_tr': 2048,
    'batch_size': 2048,
    'N_epoch': 501,
    'learning_rate': 0.01,
    'momentum': 0.9,
}

test_param_configs = {
    'gpu_id': '2',
    'n_tr_list': [1300000, 1000000, 700000, 400000, 200000, 50000],
    'num_models': 10, # same as train.repeats
    'num_repeats': 10,
    'n_ev': 20000,
    'n_te': 10000,
    'test_hard': True,
    'test_soft': False,
}




