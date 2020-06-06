# CHTE
This repository is the official implementation of the paper `An Online Sequential Test for Heterogeneous Treatment Effects` submitted to NeurIPS 2020. 

## Requirements
- Python version: Python 3.6.8 :: Anaconda custom (64-bit)
### Main packages for the proposed estimator
- numpy == 1.18.1
- pandas == 1.0.3
- scipy == 1.4.1
- statsmodels == 0.11.0
- bspline == 0.1.1
### Additional packages for experiments
- math
- os
- sys
- itertools
- multiprocessing

## Reproduce simulation results
### Synthetic data
- move to `simulation` folder, and separately run all the python scripts, i.e. ```python SABC_nonlinear_HTE.py &```, this will take several minutes, we have provided all our results as pickled files in `simulation/result/` folder as `ATE.pkl` and `HTE.pkl`.
- then open ```post_processing.ipynb```, run all the cells, we will obtain all the tables and figures in our paper.
