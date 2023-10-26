# Import
import json
import sys, os, inspect
current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))) 
sys.path.append(current_dir+'/..')
import global_config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lfi


def plot_pval_method_ntr(**kwargs):
    plt.rcParams['lines.markersize'] = 9

    df = pd.DataFrame(columns=['label', 'n_tr', 'E[pval]', 'std[pval]', 'color', 'linestyle', 'key', 'soft_or_hard'])
    n_tr_list = global_config.test_param_configs['n_tr_list']
    num_models = global_config.test_param_configs['num_models']
    num_repeat = global_config.test_param_configs['num_repeat']
    if 'n_tr_list' in kwargs:
        n_tr_list = json.loads(kwargs['n_tr_list'])

    def not_outlier(A, axis=0):
        mask = np.logical_and(np.abs(A)>0.5, np.abs(A)<10)
        mean = np.mean(A, axis, keepdims=True, where=mask)
        std = np.std(A, axis, keepdims=True, where=mask)
        scores = (A-mean)/std
        output = np.logical_and(np.abs(scores)<1.5, mask)
        return output
    def key_to_label_color_linestyle_width_marker(key, soft_or_hard=None):
        if key == 'Mix': 
            if soft_or_hard == 'soft': 
                return 'MMD-M', 'k', '-', 2, 'o'
            elif soft_or_hard == 'hard': 
                return 'MMD-M with $t_{opt}$', 'C1', '-', 3, 'o'
        if key == 'Scheffe':
            if soft_or_hard == 'soft': 
                return 'SCHE soft', 'r', '--', 2, '^'
            elif soft_or_hard == 'hard': 
                return 'SCHE with $t_{opt}$' , 'red', ':', 2, '^'
            elif type(soft_or_hard) == float: 
                return 'SCHE with t=0.5', 'b', ':', 2, '^'
        if key == 'Fea_Gau': 
            if soft_or_hard == 'soft':
                return 'MMD-G', 'm', '-.', 2, 's'
            # elif soft_or_hard == 'hard': 
            #     return 'MMD-G with $t_{opt}$', 'C0', '-.', 2, 'o'
        if key == 'Gaussian': 
            if soft_or_hard == 'soft':
                return 'MMD-O', 'y', '-', 2, 's'
        if key == 'LBI': 
            if soft_or_hard == 'soft':
                return 'LBI', 'green', '-', 2, '^'
        if key == 'UME': 
            if soft_or_hard == 'soft':
                return 'UME', 'C0', '--', 2, 'P'
        if key == 'RFM': 
            if soft_or_hard == 'soft':
                return 'RFM as classifier', 'c', '-', 2, 'X'
            elif soft_or_hard == 'as_MMD':
                return 'RFM for MMD', 'brown', '--', 2, 'X'
        return '--Skip--', None, None, None, None

    display_label_list = ['MMD-M with $t_{opt}$',
                        'MMD-M',
                        'MMD-G',
                        'MMD-O',
                        'UME',
                        'SCHE with $t_{opt}$',
                        'SCHE with t=0.5',
                        'LBI',
                        'RFM as classifier',
                        'RFM for MMD',]

    for key in global_config.method_configs:
        if global_config.method_configs[key]:
            for i_trick, trick in enumerate(['soft', 'hard', 0.5, 'as_MMD']):
                label, color, linestyle, width, marker = key_to_label_color_linestyle_width_marker(key, trick)

                if label in display_label_list:
                    save_pval_data_name = 'pval_%s.npy'%(['orig', 't_opt', 'force_t', 'as_mmd'][i_trick])
                    ckpts_dir = os.path.join(current_dir, '..', 'methods', key, 'checkpoints')
                    p_mat = np.zeros((len(n_tr_list), len(num_models), len(num_repeat)))
                    for i_n, n in enumerate(n_tr_list):
                        for r in num_models:
                            ckpt_dir = os.path.join(ckpts_dir, 'n_tr=%d#%d'%(n,r))
                            if os.path.exists(os.path.join(ckpt_dir, save_pval_data_name)):
                                p_mat[i_n,r,:] = np.load(os.path.join(ckpt_dir, save_pval_data_name))

                    print('----- method =', key, ', original -----')
                    not_outlier_mask = not_outlier(p_mat, axis=1)
                    E_pval = np.array([np.mean(p_mat[i], where=not_outlier_mask[i]) for i in range(len(n_tr_list))])
                    std_pval = np.array([np.std(p_mat[i], where=not_outlier_mask[i]) for i in range(len(n_tr_list))])
                    outlier_every_n = np.sum(1-not_outlier_mask, axis=2)
                    if np.sum(outlier_every_n)>0:
                        print('Outliers number at each (n_tr, num_model): ')
                        print('{:<{}}'.format('', 10), end="| ")
                        for j in range(outlier_every_n.shape[1]):
                                print('#', f'{j:<1}', end="| ")
                        print()
                        for i in range(outlier_every_n.shape[0]):
                            print('{:<{}}'.format(n_tr_list[i], 10), end="| ")
                            for j in range(outlier_every_n.shape[1]):
                                print(f'{outlier_every_n[i][j]:<3}', end="| ")
                            print()
                        # print(outlier_every_n)
                    new_row_soft = {'label':label,
                                    'n_tr': [n_tr_list], 
                                    'E[pval]': [E_pval],
                                    'std[pval]': [std_pval],
                                    'color': color,
                                    'linestyle':linestyle,
                                    'width':width,
                                    'marker':marker,
                                    'key': key,
                                    'soft_or_hard': trick}
                    # df = df.append(new_row_soft, ignore_index=True)
                    df = pd.concat([df, pd.DataFrame.from_dict(new_row_soft)], ignore_index=True)
                    print('new row appended')
    
    print('\n-------------------')
    print('The built dataframe =')
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)  
    print(df)
    np.set_printoptions(precision=2, suppress=True)
    pd.set_option('display.float_format', '{:.2f}'.format)
    # print(df[['label', 'E[pval]']].to_string())

    fig = plt.figure(figsize=(6, 6.6))
    for label in display_label_list:
        row = df[df['label']==label].iloc[0]
    # for index, row in df.iterrows():
        # if row['label'] in display_label_list:
        mean_list = row['E[pval]']
        std_list = row['std[pval]']
        labal = row['label']
        color = row['color']
        linestyle = row['linestyle']
        width = row['width']
        marker = row['marker']
        plt.plot(n_tr_list, mean_list, label=labal, marker=marker, alpha=0.99, color=color, linestyle=linestyle, lw=width)
        plt.fill_between(n_tr_list, mean_list-std_list, mean_list+std_list, alpha=0.2, color=color)

    plt.axhline(y=5, color='purple', linestyle='--', lw=3)

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=2)

    plt.xlabel('Training set size 2n/million', size=15)
    plt.xticks(n_tr_list, np.array(n_tr_list)*2/10**6, rotation=90)
    plt.tick_params(axis='x', which='major')
    plt.xlim(0.0*10**6/2, 2.7*10**6/2)

    # set tick precision 0.5 ,yaxis.set_major_locator(MaxNLocator(5))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
    plt.ylabel('Significance of discovery/σ', size=15)
    plt.ylim(0.6, 5.4)

    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(current_dir, '..', 'assets', 'Significance of discovery.pdf'))
    print('\n ----- DONE ----- ')
    print('The plot is saved at', os.path.join(current_dir, '..', 'assets', 'Significance of discovery.pdf'))
    print()
    

if __name__ == "__main__":
    if sys.argv[1] == 'pval':
        method = sys.argv[2]
        if method == 'plot':
            plot_pval_method_ntr(**dict([arg.split('=') for arg in sys.argv[3:]]))
        elif method == 'ALL':
            for key in global_config.method_configs:
                if global_config.method_configs[key]:
                    print('----- method =', key, '-----')
                    config_dir = os.path.join(current_dir, '..', 'methods', key)
                    lfi.test.main_pval(config_dir, **dict([arg.split('=') for arg in sys.argv[3:]]))
        elif method == 'RFM':
            os.system('python '+os.path.join(current_dir,'..','methods','RFM','RFM_test_used_in_MMD.py') +' '+ ' '.join(sys.argv[3:]))
            os.system('python '+os.path.join(current_dir,'..','methods','RFM','RFM_test.py') +' '+ ' '.join(sys.argv[3:]))
        elif method == 'Mix':
            config_dir = os.path.join(current_dir, '..', 'methods', method)
            os.system('python %s %s'%(os.path.join(config_dir, 'MMD_M_test.py'), ' '.join(sys.argv[3:])))
            lfi.test.main_pval(config_dir, **dict([arg.split('=') for arg in sys.argv[3:]]))
        else:
            config_dir = os.path.join(current_dir, '..', 'methods', method)
            lfi.test.main_pval(config_dir, **dict([arg.split('=') for arg in sys.argv[3:]]))
                
    elif sys.argv[1] == 'error':
        method = 'Mix'
        config_dir = os.path.join(current_dir, '..', 'methods', method)
        lfi.test.main_error(config_dir, **dict([arg.split('=') for arg in sys.argv[2:]]))
    else:
        print('Please enter: python test_methods.py x y z')
        print('x = pval, error, plot')
        print('y = method name, or ALL')
        print('z = other arguments, such as gpu=4 n_tr_list=[100000,200000]')

    
        
    