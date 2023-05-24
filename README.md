# Higgs-boson Discovery Experiment.

This repository contains a Python 3.9 implementation of the kernel tests described in our paper (removed for submission).

# How to install?
You don't need to install the package to reproduce our results. Nevertheless the package can be installed with the pip command `pip install .`

Once installed, you should be able to do `import lfi`.

# Dependency
Our code relies on the following Python packages. It is recommended to use the specified version numbers for optimal compatibility:

    python==3.9.12
    numpy==1.22.3
    torch==1.13.1
    importlib==1.0.4
    scipy==1.7.3
    tqdm==4.64.1
    pandas==1.5.2
    pyroc==0.1.1
    matplotlib==3.6.2
    requests==2.28.1
    IPython==8.8.0
    jupyterlab==3.5.2

Moreover our GPU is NVIDIA Quadro RTX 8000. 

# Reproduce our results

## 0. TLDR:
First, go to `global_config.py`, modify `train_param_configs['gpu']` and `test_param_configs['gpu']` to the GPU numbers that are valid on your machine.
Then run the following commands successively
```
python datasets/Download_Higgs_Dataset.py
python experiments/train_methods.py ALL
python experiments/test_methods.py pval ALL
python experiments/test_methods.py error Mix
python experiments/test_methods.py pval plot
python experiments/error_of_var_kernel.py

```
Run the Jupyter notebooks in `./experiments` the get all of the figures.
Then you will see the figures in `./assets`. Note that this process may take a few days to complete. 

The following details explain what the commands above are doing. You may also reduce number of independent runs to obtain results more quickly.
 
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
    LBI: LBI
    RFM: RFM
    UME: UME

For each `x` in `{Fea\_Gau, Gaussian, Mix, Scheffes, LBI, RFM, UME}`, there are two files, `config.py` and `model.py`, in `./methods/x`. The file `./methods/x/model.py` defines the neural network architecture and the functions for computing the loss function and test statistics. The file `./methods/x/config.py` defines the training and testing parameters. The file `./methods/x/config.py` inherits from `./global_config.py` except the `RFM` method (since `RFM` does not use gradient descent for training).

To train one of the methods, or train all methods, run 
```
python experiments/train_methods.py x kwargs
```
where `x` is in `{ALL, Fea\_Gau, Gaussian, Mix, Scheffes, LBI, RFM, UME}`, and `kwargs` can be used to temporarily change the parameters in `./methods/x/config.py`. For example, we can do (space used as a delimiter):
```
python experiments/train_methods.py Gaussian gpu=7 n_tr_list=[1000000,400000,200000] repeat=[0,1,2,3,4]
```
The trained model will be saved to `./methods/x/checkpoints/n_tr=y#z/kernel.pt`, where $y=n_{tr}$ and z=number of independent run.

*Note 1*: Plotting some figures below does not depend on all of the generated checkpoints. You can plot some of the figures before all training are finished.

## 3. Generate estimated p-values:
Run 
```
python experiments/test_methods.py pval ALL
```
to generate p-values. This generates data that are needed to plot our Figure 3. The generated data will be saved as `.npy` files in the same directory as the checkpoints `kernel.pt`. Similarly you are free to adjust parameters, like
```
python experiments/test_methods.py pval Scheffe gpu=7
```

## 4. Generate estimated test error:
Run 
```
python experiments/test_methods.py error
```
This generates data that are needed to plot our Figure 1.

## 5. Plot Figure 3:
Run 
```
python experiments/test_methods.py plot
```
You will see the plot in `./assets/Significance of discovery`.

## 6. Plot the invariant mass distribution plot in our appendix:
Open `./experiments/bin_plot_invariant_mass.ipynb` and run it.

## 7. Plot the $(m,n_{ev})$ trade-off for a fixed kernel:
Open `./experiments/trade_off_fix_kernel.ipynb` and run it.

## References
<a id="1">[1]</a> 
Pierre Baldi, Peter Sadowski, Daniel Whiteson,
Searching for Exotic Particles in High-energy Physics with Deep Learning.
Nature Communications 5 (July 2, 2014).