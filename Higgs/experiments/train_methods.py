# Import
import sys, os, inspect
current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))) 
sys.path.append(current_dir+'/..')
import lfi
import global_config

# Iter through methods
if __name__ == '__main__':
    method = sys.argv[1]
    if method == 'ALL':
        for key in global_config.method_configs:
            if global_config.method_configs[key]:
                print('----- method =', key, '-----')
                config_dir = os.path.join(current_dir, '..', 'methods', key)
                if key == 'RFM':
                    os.system('python %s %s'%(os.path.join(config_dir, 'RFM_train.py'), ' '.join(sys.argv[2:])))
                elif key == 'Mix':
                    os.system('python %s %s'%(os.path.join(config_dir, 'MMD_M.py'), ' '.join(sys.argv[2:])))
                    lfi.train.main(config_dir, **dict([arg.split('=') for arg in sys.argv[2:]]))
                else:
                    lfi.train.main(config_dir, **dict([arg.split('=') for arg in sys.argv[2:]]))

    else:
        config_dir = os.path.join(current_dir, '..', 'methods', method)
        if method == 'RFM':
            os.system('python %s %s'%(os.path.join(config_dir, 'RFM_train.py'), ' '.join(sys.argv[2:])))
        elif method == 'Mix':
            os.system('python %s %s'%(os.path.join(config_dir, 'MMD_M.py'), ' '.join(sys.argv[2:])))
            lfi.train.main(config_dir, **dict([arg.split('=') for arg in sys.argv[2:]]))
        else:
            lfi.train.main(config_dir, **dict([arg.split('=') for arg in sys.argv[2:]]))
