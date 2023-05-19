
cd /math/home/eruisun/github/LFI/trade-off_var_kernel
conda activate LFI
python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../')
import lfi
import numpy as np

ns_total = (10**np.linspace(3, 6.2, 10)).astype(int)
# ns_total = ns_total[ns_total>135930]
ns_total = [200000]
# current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.getcwd()
config_path = current_dir+'/../methods/Res_Net/config.py'
lfi.train.main(config_path, 
                n_tr_list=ns_total, 
                checkpoints_path=current_dir+'/checkpoints',
                batch_size=8192,
                gpu='1')

cd /math/home/eruisun/github/LFI
conda activate LFI
python
import sys, os
import lfi
config_path = './methods/Res_Net/config.py'
# lfi.train.main(config_path, n_tr_list=range(200, 220), batch_size=220 ,gpu='6',patience=10)

lfi.train.main(config_path, n_tr_list=[200003, 200013], batch_size=10002 ,gpu='1',patience=10)

conda activate LFI
cd /math/home/eruisun/github/LFI/trade-off_var_kernel
python Test_each_ckpt.py 1 1600000

