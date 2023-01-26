# Higgs-boson Discovery Experiment.
## General reminder.

## Preparing Dataset.
Run 'Download_Higgs_Dataset.py' to download the dataset. (Alternatively, manually download the Higgs dataset from http://archive.ics.uci.edu/ml/datasets/HIGGS and then turn the HIGGS.csv.gz file into HIGGS.npy with shape (11000000,29).)
## Training:  
Six folders correspond to different architectures.  
Fea_Gau is MMD-G.  
Gaussian is MMD-O.  
Mix is MMD-M with $\varphi'=0$.  
Res_Net is MMD-M with activated $\varphi'$.    
Scheffe is SCHE.  
LBI is LBI.    
To train one of them, open one of these folders, then run 'X.py' where X is one of MMD-M, MMD-O, MMD-G, Scheffe, LBI.    
## Plotting:  
invariant_mass.ipynb for the invariant mass distribution plot in our appendix.
trade_off_fix_kernel.ipynb for the $(m,n_{te})$ trade-off of a fixed kernel in our appendix.
generate_p_data.ipynb for calculating p-value from saved checkpoints.  
p_n_plot.ipynb for generating the p-value-n curve for different methods.  

(Notice: in the code, we switch the definition of $n_{eval}$ and $n_{test}$, i.e. everything with subscript 'eval'/'test' in our code refer to 'test'/'eval' in our paper.)