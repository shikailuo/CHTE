import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from math import sqrt, log, exp, cos, pi
from itertools import product
import multiprocessing as mp
import bspline
import bspline.splinelab as splinelab
import os, sys
path = os.path.join('..', 'src')
if path not in set(sys.path):
    sys.path.append(path)
from sab import falpha, AVT

def phi(x, delta, typ):
    if typ == 1:
        return delta*x
    elif typ == 2:
        return delta*np.cos(x/pi)
    elif typ == 3:
        return delta*x**2/3.
    else:
        return x

np.random.seed(1234567)

nsim = 400
B = 10000
start_n = 500 # sample size at first analysis
typ_list = [2, 3] # phi function type list
nk_list = [(100, 5), (10, 50)] #
delta_list = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30] # effect size typ three
alpha = 0.05
e = 1 # error variance

lb, ub = -2, 2

nfeatures = 3
p = nfeatures+1

rho = 0.5
cov_mat = (1-rho) * np.eye(nfeatures) + rho*np.ones((nfeatures, nfeatures))
L = np.linalg.cholesky(cov_mat).T

for (typ, (n, k), delta) in product(typ_list, nk_list, delta_list):
    
    name = 'result/ATE/'+'typ'+str(typ)+'n'+str(n)+'k'+str(k)+'delta'+str(delta)+'_nonlinear_AVT.npz'
        
    def one_step(seed):
        np.random.seed(seed)
        
        sabc = AVT(alpha=alpha)
        
        for j in range(k):
            this_n = start_n if j == 0 else n
            
            X0 = np.random.randn(this_n, nfeatures).dot(L).clip(lb, ub)
            X1 = np.random.randn(this_n, nfeatures).dot(L).clip(lb, ub)
            y0 = 1 + (X0[:,0]-X0[:,1])/2+ np.random.randn(this_n)*e
            y1 = 1 + (X1[:,0]-X1[:,1])/2 + phi((X1[:,0]+X1[:,1])/sqrt(2.), delta, typ)*(X1[:,2])**2 + np.random.randn(this_n)*e
            
            sabc.fit(y0, y1)
            
            if sabc.rej: break
                
        return [sabc.rej, sabc.rej, sabc.rej]
    
    pool = mp.Pool(10)
    rets = pool.map(one_step, range(nsim))
    rets = np.array(rets)
    pool.close()
    
    np.savez(name, rej_sabc=rets)