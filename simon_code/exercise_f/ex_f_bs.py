#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:00:09 2022

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
n_bootstrap = 10

##create data and designe matrix 

x,y,fval = fnc.makedata(points,sigma,fnc.FrankeFunction)
fnc.rescale = True
variables=[x,y]
A = fnc.DM(variables,max_order)
#split data into test & train
A_train, A_test, f_train, f_test = train_test_split(A, fval, test_size=0.2, random_state = 1)


##Create all needed arrays 

#Lasso Parameters
lambdas = np.zeros(9)
lambdas[:8] = np.power(10.0,2-np.arange(8))

MSE_bs = np.zeros((len(lambdas),max_order))



for i in range(1,max_order+1):
    #current number of tearms also via complete homogeneous symmetric polynomials
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)
    #select only the terms of the full desinge matrix needed for the current order
    ATrCur = A_train[:,0:currentnot]
    ATeCur = A_test[:,0:currentnot]
    k = 0
    for ridge_par in lambdas: 
        fnc.param = ridge_par
        MSE_bs[k,i-1] = fnc.bootstrap(n_bootstrap,ATrCur,ATeCur,f_train,f_test,fnc.Lasso)[0]
        k += 1

#get Index for minimum MSE
min_bs_ind = divmod(MSE_bs.argmin(), MSE_bs.shape[1])

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(MSE_bs.T, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'},fmt='.1e')
ax.set_title("Test Accuracy BSE 2")
ax.set_ylabel("order")
ax.set_xlabel("log$_{10}(\lambda)$")
ax.set_xticklabels(np.log10(lambdas,out=np.zeros_like(lambdas), where=(lambdas!=0)))
ax.set_yticklabels(range(1,max_order+1))
ax.add_patch(plt.Rectangle((min_bs_ind[0], min_bs_ind[1]), 1, 1, fc='none', ec='red', lw=2, clip_on=False))

print(f"minimum error BS: {np.min(MSE_bs):.2e}")
print(f"for lambda: {lambdas[min_bs_ind[0]]:.1e} and order: {min_bs_ind[1]}")
