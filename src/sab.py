import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import root
from math import sqrt, log, exp

eps = 1e-6

def falpha(r, alpha=0.05, method=2, theta=0.5, gamma=1):
    if method == 2:
        z_alpha_2 = norm.ppf(1-alpha/2)
        return 2-2*norm.cdf(z_alpha_2/r**0.5)
    elif method == 1:
        return alpha*np.log(1+(np.e-1)*r)
    elif method == 3:
        return alpha*(r**theta)
    else:
        return alpha*(1-np.exp(-gamma*r))/(1-np.exp(-gamma))
    
##############################################################################################################################    
class AVT(object):
    # always valid test
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.nx, self.ny = 0, 0
        self.xcs, self.ycs = 0, 0
        self.x2cs, self.y2cs = 0, 0
        self.rej = 0
        self.k = 0
    
    def fit(self, x, y, tau=1):
        self.nx += len(x)
        self.ny += len(y)
        self.xcs += x.sum()
        self.ycs += y.sum()
        self.x2cs += (x**2).sum()
        self.y2cs += (y**2).sum()
        
        self.theta = (self.ycs/self.nx - self.xcs/self.ny)
        vx = self.x2cs/self.nx - (self.xcs/self.nx)**2
        vy = self.y2cs/self.ny - (self.ycs/self.ny)**2
        v = vx/self.nx + vy/self.ny + 1e-6
        
        threshold = ( 2*log(1./self.alpha) - log(v/(v+tau)) ) * ( v*(v+tau)/tau )
        threshold = sqrt(threshold)
        
        self.k += 1
        if self.theta > threshold:
            self.rej = self.k
            
###########################################################################################################################
class SABC_ATE(object):
    # sequential AB test with covariates
    def __init__(self, alpha=0.05, B=10000, p=2):
        '''
        B: number of bootstrap samples     
        '''
        self.alpha_spent = 0.
        self.alpha = alpha
        self.B = B
        self.p = p # number of dimensions for covariates including intercept        
        self.I = np.array([True for _ in range(self.B)]) # bootstrap samples 
        self.betab0, self.betab1 = np.zeros((p, B)), np.zeros((p, B))
        
        self.xcs = np.zeros(p)
        
        self.n0, self.n1 = 0, 0 # number of samples seen for A = 0, 1 groups
        self.Sigma0, self.Sigma1 = np.zeros((p, p)), np.zeros((p, p))  
        self.invSigma0, self.invSigma1 = np.zeros((p, p)), np.zeros((p, p)) 
        self.gamma0, self.gamma1 = np.zeros(p), np.zeros(p)
        self.beta0, self.beta1 = np.zeros(p), np.zeros(p)
        
        self.k, self.rej = 0, 0        
        self.T, self.ck = [], []
        
    def fit(self, X0, y0, X1, y1, alpha, l2=1e-3, eps=1e-6, verbose=False, approx=False): # alpha should be spent until here
        B, p = self.B, self.p
        # step (a)
        phi0, phi1 = np.zeros((p,p)), np.zeros((p,p))
        # step (b)
        if y0 is not None:
            y0 = y0.flatten()
            n0 = float(len(y0))
            self.n0 += n0
            X0X0 = (X0[:,np.newaxis,:] * X0[:,:,np.newaxis]) # n0 x p x p
            X0y0 = (X0 * y0[:,np.newaxis]) # n0 x p
            self.Sigma0 = (self.n0-n0)/self.n0*self.Sigma0 + X0X0.sum(axis=0)/self.n0 # p x p
            self.gamma0 = (self.n0-n0)/self.n0*self.gamma0 + X0y0.sum(axis=0)/self.n0 # p
            self.invSigma0 = np.linalg.inv(self.Sigma0+l2*np.diag(np.ones(p)))
            
            self.beta0 = self.invSigma0.dot(self.gamma0)

            resid0 = y0 - X0.dot(self.beta0)
            phi0 = ((resid0[:,np.newaxis,np.newaxis]**2) * X0X0).sum(axis=0) # p x p
            phi0 = self.invSigma0.dot(phi0).dot(self.invSigma0)
            eigval0, eigvec0 = np.linalg.eig(phi0)
            eigval0[eigval0 < eps] = eps
            eigval0 = eigval0**0.5
            eigvec0 = eigvec0.real
            phi0 = (eigvec0*eigval0[np.newaxis,:]).dot(eigvec0.T)

            self.betab0 = (1.0-n0/self.n0)*self.betab0 + \
                          (1.0/self.n0)*phi0.dot(np.random.randn(p,B))
            self.xcs += X0.sum()
            
        if y1 is not None:
            y1 = y1.flatten()
            n1 = float(len(y1))
            self.n1 += n1
            X1X1 = (X1[:,np.newaxis,:] * X1[:,:,np.newaxis]) # n1 x p x p
            X1y1 = (X1 * y1[:,np.newaxis]) # n1 x p
            self.Sigma1 = (self.n1-n1)/self.n1*self.Sigma1 + X1X1.sum(axis=0)/self.n1 # p x p
            self.gamma1 = (self.n1-n1)/self.n1*self.gamma1 + X1y1.sum(axis=0)/self.n1 # p
            self.invSigma1 = np.linalg.inv(self.Sigma1+l2*np.diag(np.ones(p)))
            self.beta1 = self.invSigma1.dot(self.gamma1)

            resid1 = y1 - X1.dot(self.beta1)
            phi1 = ((resid1[:,np.newaxis,np.newaxis]**2) * X1X1).sum(axis=0) # p x p
            phi1 = self.invSigma1.dot(phi1).dot(self.invSigma1)
            eigval1, eigvec1 = np.linalg.eig(phi1)
            eigval1[eigval1 < eps] = eps
            eigval1 = eigval1**0.5
            eigvec1 = eigvec1.real
            phi1 = (eigvec1*eigval1[np.newaxis,:]).dot(eigvec1.T)
            
            self.betab1 = (1.0-n1/self.n1)*self.betab1 + \
                          (1.0/self.n1)*phi1.dot(np.random.randn(p,B))
            self.xcs += X1.sum()
            
        # step (c)
        T = self.xcs.dot(self.beta1-self.beta0)/(self.n0+self.n1)
        # step (d)
        Tb = self.xcs.dot(self.betab1 - self.betab0)/(self.n0+self.n1)
        # step (f)
        pt = max(eps, alpha-self.alpha_spent)/(1.0-self.alpha_spent)
        ck = Tb[self.I].mean() + Tb[self.I].std() * norm.ppf(1-pt) if approx else np.quantile(Tb[self.I], 1-pt)
    
        if verbose:
            print('k={}'.format(self.k), \
                  'alpha={:.5f}'.format(alpha), \
                  'alpha_spent={:.5f}'.format(self.alpha_spent), \
                  'pt={:.5f}'.format(pt), \
                  'norm.ppf={:.5f}'.format(norm.ppf(1-pt)))
            
        self.k += 1
        self.I = self.I & (Tb <= ck)
        if T > ck: self.rej = self.k

        self.T.append(T)
        self.ck.append(ck)
        self.alpha_spent = 1.0-self.I.mean()      
        
        
class SABC_HTE(object):
    # sequential AB test with covariates
    def __init__(self, alpha=0.05, B=10000, p=2):
        '''
        B: number of bootstrap samples     
        '''
        self.alpha_spent = 0.
        self.alpha = alpha
        self.B = B
        self.p = p # number of dimensions for covariates including intercept        
        self.I = np.array([True for _ in range(self.B)]) # bootstrap samples 
        self.betab0, self.betab1 = np.zeros((p, B)), np.zeros((p, B))
        
        self.lb, self.ub = np.zeros(p-1), np.zeros(p-1)
        
        self.n0, self.n1 = 0, 0 # number of samples seen for A = 0, 1 groups
        self.Sigma0, self.Sigma1 = np.zeros((p, p)), np.zeros((p, p))  
        self.invSigma0, self.invSigma1 = np.zeros((p, p)), np.zeros((p, p)) 
        self.gamma0, self.gamma1 = np.zeros(p), np.zeros(p)
        self.beta0, self.beta1 = np.zeros(p), np.zeros(p)
        
        self.k, self.rej = 0, 0        
        self.T, self.ck = [], []

    def fit(self, X0, y0, X1, y1, alpha, l2=1e-3, eps=1e-6, verbose=False, approx=False): # alpha should be spent until here
        B, p = self.B, self.p
        # step (a)
        phi0, phi1 = np.zeros((p,p)), np.zeros((p,p))
        # step (b)
        if y0 is not None:
            y0 = y0.flatten()
            n0 = float(len(y0))
            self.n0 += n0
            X0X0 = (X0[:,np.newaxis,:] * X0[:,:,np.newaxis]) # n0 x p x p
            X0y0 = (X0 * y0[:,np.newaxis]) # n0 x p
            self.Sigma0 = (self.n0-n0)/self.n0*self.Sigma0 + X0X0.sum(axis=0)/self.n0 # p x p
            self.gamma0 = (self.n0-n0)/self.n0*self.gamma0 + X0y0.sum(axis=0)/self.n0 # p
            self.invSigma0 = np.linalg.inv(self.Sigma0+l2*np.diag(np.ones(p)))
            
            self.beta0 = self.invSigma0.dot(self.gamma0)

            resid0 = y0 - X0.dot(self.beta0)
            phi0 = ((resid0[:,np.newaxis,np.newaxis]**2) * X0X0).sum(axis=0) # p x p
            phi0 = self.invSigma0.dot(phi0).dot(self.invSigma0)
            eigval0, eigvec0 = np.linalg.eig(phi0)
            eigval0[eigval0 < eps] = eps
            eigval0 = eigval0**0.5
            eigvec0 = eigvec0.real
            phi0 = (eigvec0*eigval0[np.newaxis,:]).dot(eigvec0.T)
            
            self.betab0 = (1.0-n0/self.n0)*self.betab0 + \
                          (1.0/self.n0)*phi0.dot(np.random.randn(p,B))
            
            self.lb = np.minimum(self.lb, X0[:,1:].min(axis=0))
            self.ub = np.maximum(self.ub, X0[:,1:].max(axis=0))
            
        if y1 is not None:
            y1 = y1.flatten()
            n1 = float(len(y1))
            self.n1 += n1
            X1X1 = (X1[:,np.newaxis,:] * X1[:,:,np.newaxis]) # n1 x p x p
            X1y1 = (X1 * y1[:,np.newaxis]) # n1 x p
            self.Sigma1 = (self.n1-n1)/self.n1*self.Sigma1 + X1X1.sum(axis=0)/self.n1 # p x p
            self.gamma1 = (self.n1-n1)/self.n1*self.gamma1 + X1y1.sum(axis=0)/self.n1 # p
            self.invSigma1 = np.linalg.inv(self.Sigma1+l2*np.diag(np.ones(p)))
            
            self.beta1 = self.invSigma1.dot(self.gamma1)

            resid1 = y1 - X1.dot(self.beta1)
            phi1 = ((resid1[:,np.newaxis,np.newaxis]**2) * X1X1).sum(axis=0) # p x p
            phi1 = self.invSigma1.dot(phi1).dot(self.invSigma1)
            eigval1, eigvec1 = np.linalg.eig(phi1)
            eigval1[eigval1 < eps] = eps
            eigval1 = eigval1**0.5
            eigvec1 = eigvec1.real
            phi1 = (eigvec1*eigval1[np.newaxis,:]).dot(eigvec1.T)
            
            self.betab1 = (1.0-n1/self.n1)*self.betab1 + \
                          (1.0/self.n1)*phi1.dot(np.random.randn(p,B))
            
            self.lb = np.minimum(self.lb, X1[:,1:].min(axis=0))
            self.ub = np.maximum(self.ub, X1[:,1:].max(axis=0))
            
        # step (c)
        T = (self.beta1-self.beta0)
        T0, T1 = T[0], T[1:]
        T = T0 + (T1*(T1>0)*self.ub).sum() + (T1*(T1<0)*self.lb).sum()
        # step (d)
        Tb = (self.betab1 - self.betab0)
        Tb0, Tb1 = Tb[0], Tb[1:]
        Tb = Tb0 + (Tb1*(Tb1>0)*self.ub[:,np.newaxis]).sum(axis=0) + (Tb1*(Tb1<0)*self.lb[:,np.newaxis]).sum(axis=0)
        # step (f)
        pt = max(eps, alpha-self.alpha_spent)/(1.0-self.alpha_spent)
        ck = Tb[self.I].mean() + Tb[self.I].std() * norm.ppf(1-pt) if approx else np.quantile(Tb[self.I], 1-pt)
        
        if verbose:
            print('k={}'.format(self.k), \
                  'alpha={:.5f}'.format(alpha), \
                  'alpha_spent={:.5f}'.format(self.alpha_spent), \
                  'pt={:.5f}'.format(pt), \
                  'norm.ppf={:.5f}'.format(norm.ppf(1-pt)))
            
        self.k += 1
        self.I = self.I & (Tb <= ck)
        if T > ck: self.rej = self.k
        
        self.T.append(T)
        self.ck.append(ck)
        self.alpha_spent = 1.0-self.I.mean()        

        
class SABC_LIL(object):
    # sequential AB test with covariates
    def __init__(self, p=2):
        self.p = p # number of dimensions for covariates including intercept        
        
        self.xnorm = 0
        
        self.n = 0
        
        self.Sigma0, self.Sigma1 = np.zeros((p, p)), np.zeros((p, p))  
        self.invSigma0, self.invSigma1 = np.zeros((p, p)), np.zeros((p, p))
        self.gamma0, self.gamma1 = np.zeros(p), np.zeros(p)
        self.beta0, self.beta1 = np.zeros(p), np.zeros(p)
        
        self.lb, self.ub = np.zeros(p-1), np.zeros(p-1)
        
        self.k, self.rej = 0, 0        
        self.T, self.ck = [], []
        self.c = 0

    def fit(self, X0, y0, X1, y1, l2=1e-3): # alpha should be spent until here
        p = self.p
        # compute sample size here
        n = 0
        n += len(y0) if y0 is not None else 0
        n += len(y1) if y1 is not None else 0
        self.n += n
        
        this_c = 0
        if y0 is not None:
            y0 = y0.flatten()
            X0X0 = (X0[:,np.newaxis,:] * X0[:,:,np.newaxis]) # n0 x p x p
            X0y0 = (X0 * y0[:,np.newaxis]) # n0 x p
            
            self.Sigma0 = (1.0-n/self.n)*self.Sigma0 + X0X0.sum(axis=0)/self.n # p x p
            self.gamma0 = (1.0-n/self.n)*self.gamma0 + X0y0.sum(axis=0)/self.n # p
            self.invSigma0 = np.linalg.inv(self.Sigma0+l2*np.diag(np.ones(p))) 
            self.beta0 = self.invSigma0.dot(self.gamma0)

            resid0 = y0 - X0.dot(self.beta0)
            this_c += ( (X0.dot(self.invSigma0)**2).sum(axis=1) * (resid0**2) ).sum()
            
            self.xnorm = max(self.xnorm, (X0**2).sum(axis=1).max()**0.5 )
            
            self.lb = np.minimum(self.lb, X0[:,1:].min(axis=0))
            self.ub = np.maximum(self.ub, X0[:,1:].max(axis=0))
            
        if y1 is not None:
            y1 = y1.flatten()
            X1X1 = (X1[:,np.newaxis,:] * X1[:,:,np.newaxis]) # n1 x p x p
            X1y1 = (X1 * y1[:,np.newaxis]) # n1 x p
            
            self.Sigma1 = (1.0-n/self.n)*self.Sigma1 + X1X1.sum(axis=0)/self.n # p x p
            self.gamma1 = (1.0-n/self.n)*self.gamma1 + X1y1.sum(axis=0)/self.n # p
            self.invSigma1 = np.linalg.inv(self.Sigma1+l2*np.diag(np.ones(p)))
            self.beta1 = self.invSigma1.dot(self.gamma1)

            resid1 = y1 - X1.dot(self.beta1)
            this_c += ( (X1.dot(self.invSigma1)**2).sum(axis=1) * (resid1**2) ).sum()
            
            self.xnorm = max(self.xnorm, (X1**2).sum(axis=1).max()**0.5 )
            
            self.lb = np.minimum(self.lb, X1[:,1:].min(axis=0))
            self.ub = np.maximum(self.ub, X1[:,1:].max(axis=0))         
            
        # compute test stats
        T = (self.beta1-self.beta0)
        T0, T1 = T[0], T[1:]
        T = T0 + (T1*(T1>0)*self.ub).sum() + (T1*(T1<0)*self.lb).sum()
        
        # compute stopping boundary
        self.c += this_c
        ck = self.xnorm * np.sqrt(2*np.log(np.log(self.n))/self.n) * np.sqrt(self.c / self.n)
           
        self.k += 1
        if T > ck: self.rej = self.k
        
        self.T.append(T)
        self.ck.append(ck)