# Import
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir+'/..')
import lfi
import global_config

# Iter through methods
if __name__ == '__main__':
    for key in global_config.method_configs:
        if global_config.method_configs[key]:
            print('----- method =', key, '-----')
            lfi_train_path = os.path.join(current_dir, '..', 'lfi', 'train.py')
            os.system('python ' + lfi_train_path + key + ' ' + ' '.join(sys.argv[1:]))
            
# python lfi/train.py Mix n_tr_list=[700000,400000,200000,50000] repeat=[0] gpu=7