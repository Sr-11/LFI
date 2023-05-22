# Higgs-boson Discovery Experiment.

This repository contains a Python 3.9 implementation of the kernel tests described in our paper (removed for submission).

# How to install?
You don't need to install the package to reproduce our results. But if you want to, the package can be installed with the pip command `pip install .`

Once installed, you should be able to do `import lfi`.

# Dependency
The development of this project relied on the following Python packages. It is recommended to use the specified version numbers for optimal compatibility:

    python==3.9.12
    numpy==1.22.3
    torch==1.13.1
    importlib==1.0.4
    tqdm==4.64.1
    scipy==1.7.3
    pandas==1.5.2
    pyroc==0.1.1
    matplotlib==3.6.2
    IPython==8.8.0
    requests==2.28.1

Moreover our GPU is NVIDIA Quadro RTX 8000. 

# Reproduce our results

## 0. TLDR:
First, go to `global_config.py`, modify `train_param_configs['gpu']` and `test_param_configs['gpu']` to the GPU numbers that are valid on your machine.
Then run the following commands successively
```
python datasets/Download_Higgs_Dataset.py
python experiments/train_methods.py ALL
python experiments/test_methods.py pval ALL
python experiments/test_methods.py plot
python experiments/test_methods.py error Mix
```
Then you will see the figures in `./assets`. Note that this process may take a few days to complete. 

The following details explain what the commands above are doing. Follow the steps below to reproduce the figures and data in our paper. You may also reduce number of independent runs to obtain results more quickly.
 
## 1. Prepare dataset:
The Higgs dataset can be found at http://archive.ics.uci.edu/ml/datasets/HIGGS [[1]](#1).
Run 'python datasets/Download_Higgs_Dataset.py' to download the dataset.  This will create a file `./datasets/HIGGS.npy` whose size is about 2.4GB. 
(Alternatively, manually download the Higgs dataset from http://archive.ics.uci.edu/ml/datasets/HIGGS, turn HIGGS.csv.gz into HIGGS.npy with shape (11000000,29), then put it into `./datasets`).  

## 2. Training:  
The implementations of different benchmarks can be found in `./methods`. The correspondence between the folder name and the name in our paper is

    Fea_Gau: MMD-G
    Gaussian: MMD-O
    Mix: MMD-M
    Scheffe: SCHE
    LBI, RFM, UME: LBI, RFM, UME

For each $x\in\{\text{Fea\_Gau, Gaussian, Mix, Scheffes, LBI, RFM, UME}\}$, there are 2 files (`config.py` and `model.py`) in `./methods/x`. 
 

Run 
```
python experiments/train_methods.py ALL
```
to train. 

*Note 1*: Plotting some figures below does not depend on all of the generated checkpoints, so one can quickly plot something before all training is over. 

#### 3. Generate estimated p-values:

#### 4. Generate estimated test error:

#### 5. Plot results

### 3. Plot the invariant mass distribution plot in our appendix:
Invariant_mass.ipynb for the invariant mass distribution plot in our appendix.
### 4. Plot the $(m,n_{test})$ trade-off for different Type 1 error + Type 2 error levels for a fixed kernel:
Trade_off_fix_kernel.ipynb for the $(m,n_{test})$ trade-off of a fixed kernel in our appendix.
### 5. Plot the p-value vs n curve: 
Generate_p_data.ipynb for calculating p-value from saved checkpoints, then run p_n_plot.ipynb for generating the p-value-n curve for different methods.  
### 6. Plot the $(m,n_{train})$ trade-off for different Type 1 error + Type 2 error levels:
First run generate_error_m_n_train.ipynb 10 times following the instruction in it (one need to change $r$ in it and manually rerun it). After obtaining data, run plot_error_m_n_train.ipynb to generate the figure.

## References
<a id="1">[1]</a> 
Pierre Baldi, Peter Sadowski, Daniel Whiteson,
Searching for Exotic Particles in High-energy Physics with Deep Learning.
Nature Communications 5 (July 2, 2014).