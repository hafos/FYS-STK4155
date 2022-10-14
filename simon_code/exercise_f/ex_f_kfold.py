#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 14:53:36 2022

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
max_order = 10
#number of Folds
kfold = 4

##create data and designe matrix 

x,y,fval = fnc.makedata(points,sigma,fnc.FrankeFunction)
fnc.rescale = False
variables=[x,y]
A = fnc.DM(variables,max_order)

##Create all needed arrays 

#Lasso Parameters
lambdas = np.zeros(9)
lambdas[:8] = np.power(10.0,2-np.arange(8))

MSE_Kfold = np.zeros((len(lambdas),max_order))


for i in range(1,max_order+1):
    #current number of tearms via complete homogeneous symmetric polynomials
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)
    #select only the terms of the full desinge matrix needed for the current order
    A_curr = A[:,0:currentnot] + 0
    k = 0
    for ridge_par in lambdas: 
        fnc.param = ridge_par
        MSE_Kfold[k,i-1] = fnc.crossval(kfold,A_curr,fval,fnc.Lasso)
        k += 1
 
min_Kfold_ind = divmod(MSE_Kfold.argmin(), MSE_Kfold.shape[1])

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(MSE_Kfold.T, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'},fmt='.1e')
ax.set_title("Test Accuracy KFOLD 1")
ax.set_ylabel("order")
ax.set_xlabel("log$_{10}(\lambda)$")
ax.set_xticklabels(np.log10(lambdas,out=np.zeros_like(lambdas), where=(lambdas!=0)))
ax.set_yticklabels(range(1,max_order+1))
ax.add_patch(plt.Rectangle((min_Kfold_ind[0], min_Kfold_ind[1]), 1, 1, fc='none', ec='red', lw=2, clip_on=False))

print(f"minimum error KFOLD: {np.min(MSE_Kfold):.2e}")
print(f"for lambda: {lambdas[min_Kfold_ind[0]]:.1e} and order: {min_Kfold_ind[1]}")
