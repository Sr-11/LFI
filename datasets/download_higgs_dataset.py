import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import os, inspect

# current_dir = os.path.abspath(os.path.dirname(inspect.getfile(inspect.currentframe()))) 
current_path = os.path.dirname(os.path.realpath(__file__))
csv_file_path = os.path.join(current_path, "HIGGS.csv.gz")
npy_file_path = os.path.join(current_path, "HIGGS.npy")

print()
print('Start downloading Higgs dataset to %s...'%csv_file_path)
response = requests.get('http://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz', stream=True)
total_size = int(response.headers.get('content-length', 0))
progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
with open(csv_file_path, 'wb') as file:
    for data in response.iter_content(chunk_size=1024):
        file.write(data)
        progress_bar.update(len(data))
progress_bar.close()
print('Higgs dataset saved at %s...'%csv_file_path)

print('Start turnning it into %s...'%npy_file_path)
progress_bar = tqdm(unit=' rows', desc='Reading csv')
csv_file = pd.read_csv(csv_file_path, header=None, skiprows=lambda x: progress_bar.update(1) and False)
progress_bar.close()
np.save(npy_file_path, csv_file.values)
print('Higgs dataset saved at %s...'%npy_file_path)
print()
