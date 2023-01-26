# Higgs-boson Discovery Experiment.
Follow the instructions below.
## Train networks.
### 1. Preparing Dataset:
Run 'Download_Higgs_Dataset.py' to download the dataset.  The file size of 'HIGGS.npy' is about 2.4GB. 

(Alternatively, manually download the Higgs dataset from http://archive.ics.uci.edu/ml/datasets/HIGGS and turn the HIGGS.csv.gz file into HIGGS.npy with shape (11000000,29), then put it in the same path as this README.md).  
### 2. Training:  
Six folders correspond to different architectures in our paper.  
Folder 'Fea_Gau' is MMD-G.  
Folder 'Gaussian' is MMD-O.  
Folder 'Mix' is MMD-M with $\varphi'=0$.  
Folder 'Res_Net' is MMD-M with activated $\varphi'$.    
Folder 'Scheffe' is SCHE.  
Folder 'LBI' is LBI.    

To train one of them, one need to open one of these folders, then run 'X.py' in it where X is one of 'MMD-M,' 'MMD-O,' 'MMD-G,' 'Scheffe,' and 'LBI.' After running 'X.py,' several checkpoints (for different $n_{train}$ and different epochs) will be generated in this folder. Check if 'os.environ["CUDA_VISIBLE_DEVICES"]' in 'X.py' match with your machine first.

Note 1: Plotting some figures below does not depend on all of the generated checkpoints, so one can quickly plot something before all training is over. However, running through all of these 6 'X.py' is necessary if one want to reproduce all of our results. 

Note 2: Please ensure that you navigate to the directory where the file 'X.py' is located before running it. This will ensure that the script can successfully locate the required dataset 'HIGGS.npy'.

Note 3: For convinience, the last checkpoint during training (training is stopped once the minimum validation loss does not decrease for 5 epochs) is saved to '/checkpoint $n$/0/' (e.g. 'Res_Net/checkpoint1300000/0/')
## Calculate results.
Note 4: in our code, we refer dataset_P to the background data ($P_X$ in our paper), and dataset_Q to the signal data ($P_Y$ in our paper). Also, any variable with subscript $P$ or $Q$ correspond to $P_X$ or $P_Y$.
### 3. Plot the invariant mass distribution plot in our appendix:
invariant_mass.ipynb for the invariant mass distribution plot in our appendix.
### 4. Plot the significance vs n curve or the (m,n) trade-off for different significance levels:
trade_off_fix_kernel.ipynb for the $(m,n_{te})$ trade-off of a fixed kernel in our appendix.
### 5. Plot the p-value vs n curve: 
generate_p_data.ipynb for calculating p-value from saved checkpoints, then run p_n_plot.ipynb for generating the p-value-n curve for different methods.  
