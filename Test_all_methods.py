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
            os.system('python ' + current_dir+'/methods/'+key+'/RFM_test.py')
        elif key == 'UME':
            os.system('python ' + current_dir+'/methods/'+key+'/kmod_test.py')
        else: 
            lfi.test.main_pval(config_path)
        
