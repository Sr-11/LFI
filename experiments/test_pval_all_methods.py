# Import
from re import L
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir+'/..')
import lfi
import global_config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_all_methods():
    # Iter through methods
    for key in global_config.method_configs:
        if global_config.method_configs[key]:
            print('----- method =', key, '-----')
            lfi_train_path = os.path.join(current_dir, '..', 'lfi', 'test.py')
            os.system('python ' + lfi_train_path + key + 'pval ' + ' '.join(sys.argv[2:]))

def plot_pval_method_ntr():
    df = pd.DataFrame(columns=['label', 'n_tr', 'E[pval]', 'std[pval]', 'color', 'linestyle', 'key', 'soft_or_hard', 'display_flag'])
    n_tr_list = global_config.test_param_configs['n_tr_list']
    num_models = global_config.test_param_configs['num_models']
    num_repeat = global_config.test_param_configs['num_repeat']
    def not_outlier(A, axis=0):
        mean = np.mean(A, axis, keepdims=True)
        std = np.std(A, axis, keepdims=True)
        scores = (A-mean)/std
        return np.abs(scores)<3
    def key_to_label(key, soft_or_hard=None):
        if key == 'Mix': 
            if soft_or_hard == 'soft': return 'MMD-M'
            if soft_or_hard == 'hard': return 'MMD-M with $t_{opt}$'
        if key == 'Scheffe':
            if soft_or_hard == 'soft': return 'SCHE soft'
            if soft_or_hard == 'hard': return 'SCHE with $t=%.1f$%t' 
            if type(soft_or_hard) == float: return 'SCHE with $t_{opt}$'
        if key == 'Fea_Gau': 
            if soft_or_hard == 'soft': return 'MMD-G'
            if soft_or_hard == 'hard': return 'MMD-G with $t_{opt}$'
        if key == 'Gaussian': 
            return 'MMD-O'
        if key == 'LBI': 
            return 'LBI'
        if key == 'UME': 
            return 'UME'
        if key == 'RFM': 
            return 'RFM'
    def key_to_color(key, soft_or_hard):
        if key == 'Mix': return 'k'
        if key == 'Scheffe': return 'r'
        if key == 'Fea_Gau': return 'b'
        if key == 'Gaussian': return 'g'
        if key == 'LBI': return 'c'
        if key == 'UME': return 'm'
        if key == 'RFM': return 'y'
    def key_to_linestyle(key, soft_or_hard):
        if key == 'Mix': return '-'
        if key == 'Scheffe': return '--'
        if key == 'Fea_Gau': return '-.'
        if key == 'Gaussian': return ':'
        if key == 'LBI': return '-'
        if key == 'UME': return '-'
        if key == 'RFM': return '-'
    for key in global_config.method_configs:
        if global_config.method_configs[key]:
            ckpts_dir = os.path.join(current_dir, '..', 'methods', key, 'checkpoints')
            p_soft_mat = np.zeros((len(n_tr_list), len(num_models), len(num_repeat)))
            p_hard_mat = np.zeros((len(n_tr_list), len(num_models), len(num_repeat)))
            p_force_thres = np.zeros((len(n_tr_list), len(num_models), len(num_repeat)))
            soft_flag = True; hard_flag = True; force_flag = True
            for i_n, n in enumerate(n_tr_list):
                for r in num_models:
                    ckpt_dir = os.path.join(ckpts_dir, 'n_tr=%d#%d'%(n,r))
                    if soft_flag and os.path.exists(os.path.join(ckpt_dir, 'pval_orig.npy')):
                        p_soft_mat[i_n,r,:] = np.load(os.path.join(ckpt_dir, 'pval_orig.npy'))
                    else:
                        soft_flag = False
                    if hard_flag and os.path.exists(os.path.join(ckpt_dir, 'pval_t_opt.npy')):
                        p_hard_mat[i_n,r,:] = np.load(os.path.join(ckpt_dir, 'pval_t_opt.npy'))
                    else:
                        hard_flag = False
                    if force_flag and os.path.exists(os.path.join(ckpt_dir, 'pval_force_t.npy')):
                        p_force_thres[i_n,r,:] = np.load(os.path.join(ckpt_dir, 'pval_force_t.npy'))
                    else:
                        force_flag = False
            if soft_flag:
                print('----- method =', key, ', original -----')
                not_outlier_mask = not_outlier(p_soft_mat, axis=1)
                E_pval = np.array([np.mean(p_soft_mat[i], where=not_outlier_mask[i]) for i in range(len(n_tr_list))])
                std_pval = np.array([np.std(p_soft_mat[i], where=not_outlier_mask[i]) for i in range(len(n_tr_list))])
                outlier_every_n = np.sum(1-not_outlier_mask, axis=2)
                if np.sum(outlier_every_n)>0:
                    print('Has outlier at (n_tr, num_model) = ')
                    print(outlier_every_n)
                new_row_soft = {'label': key_to_label(key, 'soft'),
                                'n_tr': n, 
                                'E[pval]': [E_pval],
                                'std[pval]': [std_pval],
                                'color': key_to_color(key, 'soft'),
                                'linestyle': key_to_linestyle(key, 'soft'),
                                'key': key,
                                'soft_or_hard': 'soft',
                                'display_flag': True}
                # df = df.append(new_row_soft, ignore_index=True)
                print('append new row')
                df = pd.concat([df, pd.DataFrame.from_dict(new_row_soft)], ignore_index=True)
            # if hard_flag:
            #     new_row_hard = {'label': key_to_label(key, 'hard'),
            #                     'n_tr': n,
            #                     'E[pval]': [np.mean(p_hard_mat[i], where=not_outlier(p_hard_mat[i])) for i in range(len(n_tr_list))],
            #                     'std[pval]': [np.std(p_hard_mat[i], where=not_outlier(p_hard_mat[i])) for i in range(len(n_tr_list))],
            #                     'color': key_to_color(key, 'hard'),
            #                     'linestyle': key_to_linestyle(key, 'hard'),
            #                     'key': key,
            #                     'soft_or_hard': 'hard',
            #                     'display_flag': True}
            #     # df.append(new_row_hard, ignore_index=True)
            #     print('append new row')
            #     df = pd.concat([df, pd.DataFrame.from_dict(new_row_hard)], ignore_index=True)
            # if force_flag:
            #     new_row_force = {'label': key_to_label(key, 0.5),
            #                     'n_tr': n,  
            #                     'E[pval]': [np.mean(p_force_thres[i], where=not_outlier(p_force_thres[i])) for i in range(len(n_tr_list))],
            #                     'std[pval]': [np.std(p_force_thres[i], where=not_outlier(p_force_thres[i])) for i in range(len(n_tr_list))],
            #                     'color': key_to_color(key, 0.5),
            #                     'linestyle': key_to_linestyle(key, 0.5),
            #                     'key': key,
            #                     'soft_or_hard': 'hard',
            #                     'display_flag': True}
            #     # df.append(new_row_force, ignore_index=True)
            #     print('append new row')
            #     df = pd.concat([df, pd.DataFrame.from_dict(new_row_force)], ignore_index=True)
    display_label_list = ['MMD-M with $t_{opt}$',
              'MMD-M',
              'MMD-G',
              'MMD-O',
              'SCHE with $t_{opt}$',
              'SCHE with t=0.5',
              'LBI',
              'UME',
              'RFM']
    print(df)
    fig = plt.figure(figsize=(6, 6))
    for index, row in df.iterrows():
        if row['label'] in display_label_list:
            mean_list = row['E[pval]']
            std_list = row['std[pval]']
            labal = row['label']
            color = row['color']
            linestyle = row['linestyle']
            plt.plot(n_tr_list, mean_list, label=row['label'], marker='o', alpha=0.99, color=color, linestyle=linestyle)
            plt.fill_between(n_tr_list, mean_list-std_list, mean_list+std_list, alpha=0.2, color=color)

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)
    plt.xlabel('Training set size 2n/million')
    plt.xticks(n_tr_list, np.array(n_tr_list)*2/10**6)
    plt.tick_params(axis='x', which='major', labelsize=7)
    plt.xticks(rotation=90)
    plt.ylabel('Significance of discovery/σ')
    plt.ylim(0.9, 5.5)
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, '..', 'assets', 'Significance of discovery.pdf'))
    print('done')

if __name__ == '__main__':
    if sys.argv[1] == 'pval':
        test_all_methods()
    if sys.argv[1] == 'plot':
        plot_pval_method_ntr()