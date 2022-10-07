#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:40:08 2022

@author: simon
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import resample

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed 

#import all needed functions located in seperate documents
from FrankeFunction import FrankeFunction
from functions import *

np.random.seed(2018)

# Make data
datapoints = 20
sigma = 0.05
#x = np.arange(0, 1, 0.05)
#y = np.arange(0, 1, 0.05)
x = np.random.rand(datapoints)
y = np.random.rand(datapoints)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y, sigma)

##Full designe Matrix
max_order = 16
rescale = True
#Need this for the case we do not rescale -> add zero to fapproox
z_scale = 0
variables=[x,y]
numbofterms = sp.special.comb(len(variables) + max_order,max_order,exact=True)
A = DM(variables,max_order)

#Ridge Stuff
lambdas = np.zeros(9)
lambdas[:8] = np.power(10.0,2-np.arange(8))

#bootstrap
n_bootstrap = 100

#Error arrays
MSE_bs = np.zeros((len(lambdas),max_order))
BIAS = np.zeros((len(lambdas),max_order))
var = np.zeros((len(lambdas),max_order))

A_train, A_test, z_train, z_test = train_test_split(A, z, test_size=0.2)
if rescale == True:
    A_test -= np.mean(A_train,axis=0)
    A_train -= np.mean(A_train,axis=0)
    z_scale = np.mean(z_train)
    z_train -= z_scale

for i in range(1,max_order+1):
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)
    ATrCur = A_train[:,0:currentnot]
    ATeCur = A_test[:,0:currentnot]
    k = 0
    for ridge_par in lambdas: 
        beta = np.linalg.pinv(ATrCur.T @ ATrCur + ridge_par*np.identity(ATrCur.T.shape[0])) 
        beta = beta @ ATrCur.T @ z_train
        f_approx_test = np.empty((len(z_test), n_bootstrap+1))
        f_approx_test[:,0] = ATeCur @ beta + z_scale
        for j in range(1,n_bootstrap+1):
            A_res, z_res = resample(ATrCur, z_train, random_state = 1)
            beta = np.linalg.pinv(A_res.T @ A_res + ridge_par*np.identity(A_res.T.shape[0]))
            beta = beta @ A_res.T @ z_res
            f_approx_test[:,j] = ATeCur @ beta + z_scale
        BIAS[k,i-1] = np.mean( (z_test.reshape(-1, 1) - np.mean(f_approx_test, axis=1, keepdims=True))**2 )
        var[k,i-1] = np.mean( np.var(f_approx_test, axis=1, keepdims=True) )
        MSE_bs[k,i-1] = np.mean( np.mean((z_test.reshape(-1, 1) - f_approx_test)**2, axis=1, keepdims=True))
        k += 1

min_MSE_ind = divmod(MSE_bs.argmin(), MSE_bs.shape[1])
min_BIAS_ind = divmod(BIAS.argmin(), BIAS.shape[1])
min_var_ind = divmod(var.argmin(), var.shape[1])

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(MSE_bs.T, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'}, fmt='.1e')
ax.set_title("MSE")
ax.set_ylabel("order")
ax.set_xlabel("log$_{10}(\lambda)$")
ax.set_xticklabels(np.log10(lambdas,out=np.zeros_like(lambdas), where=(lambdas!=0)))
ax.set_yticklabels(range(1,max_order+1))
ax.add_patch(plt.Rectangle((min_MSE_ind[0], min_MSE_ind[1]), 1, 1, fc='none', ec='red', lw=2, clip_on=False))

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(BIAS.T, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'}, fmt='.1e')
ax.set_title("BIAS")
ax.set_ylabel("order")
ax.set_xlabel("log$_{10}(\lambda)$")
ax.set_xticklabels(np.log10(lambdas,out=np.zeros_like(lambdas), where=(lambdas!=0)))
ax.set_yticklabels(range(1,max_order+1))
ax.add_patch(plt.Rectangle((min_BIAS_ind[0], min_BIAS_ind[1]), 1, 1, fc='none', ec='red', lw=2, clip_on=False))

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(var.T, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'}, fmt='.1e')
ax.set_title("var")
ax.set_ylabel("order")
ax.set_xlabel("log$_{10}(\lambda)$")
ax.set_xticklabels(np.log10(lambdas,out=np.zeros_like(lambdas), where=(lambdas!=0)))
ax.set_yticklabels(range(1,max_order+1))
ax.add_patch(plt.Rectangle((min_var_ind[0], min_var_ind[1]), 1, 1, fc='none', ec='red', lw=2, clip_on=False))
plt.show() 

plt.scatter(np.arange(1, max_order+1, 1.0), BIAS[lambdas.argmin()], label='BIAS', color='orange', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_bs[lambdas.argmin()], label='test', color='blue', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), var[lambdas.argmin()], label='var', color='red', s=15)     
plt.legend()
plt.show()




        