# Import
import lfi
import global_config
import sys, os

# Iter through methods
for key in global_config.method_configs:
    if global_config.method_configs[key]:
        print('----- method =', key, '-----')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = current_dir+'/methods/'+key+'/config.py'
        if key == 'RFM': 
            os.system('python ' + current_dir+'/methods/'+key+'/RFM_train.py')
        elif key == 'UME':
            os.system('python ' + current_dir+'/methods/'+key+'/kmod_train.py')
        else:
            lfi.train.main(config_path)
            

# import numpy as np

# x=0.8
# y=0.8

# x/(1-y)
# (x+y-1)/np.sqrt(y*(1-y))