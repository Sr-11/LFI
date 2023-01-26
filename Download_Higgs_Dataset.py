import pandas as pd
import numpy as np
import requests
print('Start downloading Higgs dataset...')
r = requests.get('http://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz')
with open('HIGGS.csv.gz', 'wb') as f:
    f.write(r.content)
print('Higgs dataset saved as HIGGS.csv.gz...')
print('Start turnning into HIGGS.npy...')
dataset = pd.read_csv('HIGGS.csv.gz', header=None)
dataset = dataset.values
np.save('HIGGS.npy', dataset)
print('Higgs dataset saved as HIGGS.npy saved...')