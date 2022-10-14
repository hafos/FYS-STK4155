#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:31:41 2022

@author: simon
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

#Import all needed functions from different py file
from funcs import functions as fnc


#IF sigma is zero there is no noise in the Model
sigma = 0.1
#Number of x & y points, total amount of datapoints is this squared
points = 30
#Programm will fit a polnyomial up to this order
max_order = 14
#number of Folds
n_bootstrap = 10

##create data and designe matrix 

x,y,fval = fnc.makedata(points,sigma,fnc.FrankeFunction)
fnc.rescale = True
variables=[x,y]
A = fnc.DM(variables,max_order)
#split data into test & train
A_train, A_test, f_train, f_test = train_test_split(A, fval, test_size=0.2, random_state = 1)

#Lasso Parameters
lambdas = np.zeros(9)
lambdas[:8] = np.power(10.0,2-np.arange(8))

#Error arrays
MSE_bs = np.zeros((len(lambdas),max_order))
BIAS = np.zeros((len(lambdas),max_order))
var = np.zeros((len(lambdas),max_order))

for i in range(1,max_order+1):
    #current number of tearms via complete homogeneous symmetric polynomials
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)
    #select only the terms of the full desinge matrix needed for the current order
    ATrCur = A_train[:,0:currentnot] + 0
    ATeCur = A_test[:,0:currentnot] + 0
    k = 0
    for lasso_par in lambdas: 
        fnc.param = lasso_par
        MSE_bs[k,i-1], fte_aval = fnc.bootstrap(n_bootstrap,ATrCur,ATeCur,f_train,f_test,fnc.Lasso)
        BIAS[k,i-1] = np.mean( (f_test.reshape(-1, 1) - np.mean(fte_aval, axis=1, keepdims=True))**2 )
        var[k,i-1] = np.mean( np.var(fte_aval, axis=1, keepdims=True) )
        k +=1

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



        