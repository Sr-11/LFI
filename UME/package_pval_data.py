import numpy as np
import config
import sys
import pickle

ns = config.test_param_configs['n_tr_list']
pval_mid_path = config.expr_configs['pval_mat_path']

num_models = config.test_param_configs['num_models']
num_repeats = config.test_param_configs['num_repeats']

pval_dict = {}
pval_dict['UME'] = {}
pval_dict['UME']['soft'] = {}
pval_dict['UME']['hard'] = {}

for n_tr in ns:
    print("------------------- n_tr = %d -------------------"%n_tr)
    p_soft_mat = np.load(sys.path[0]+pval_mid_path+'/n_tr=%d_soft.npy'%n_tr)
    p_hard_mat = np.load(sys.path[0]+pval_mid_path+'/n_tr=%d_hard.npy'%n_tr)
    assert p_soft_mat.shape == p_hard_mat.shape == (num_models, num_repeats)
    pval_dict['UME']['soft'][n_tr] = p_soft_mat
    pval_dict['UME']['hard'][n_tr] = p_hard_mat

with open(sys.path[0]+pval_mid_path+'/pval_dict.pkl', 'wb') as f:
    pickle.dump(pval_dict, f, pickle.HIGHEST_PROTOCOL)


