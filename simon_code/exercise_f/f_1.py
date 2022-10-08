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
from sklearn import linear_model


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
sigma = 0.1
x = np.random.rand(datapoints)
y = np.random.rand(datapoints)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y, sigma)

##Full designe Matrix
max_order = 10

rescale = True
#Need this for the case we do not rescale -> add zero to fapproox
z_scale = 0
A_scale = 0

variables=[x,y]
numbofterms = sp.special.comb(len(variables) + max_order,max_order,exact=True)
A = DM(variables,max_order)

#bootstrap
n_bootstrap = 100
##Cross_val
k = 10
rs = KFold(n_splits=k,shuffle=True, random_state=1)

#Lasso Stuff
lambdas = np.zeros(9)
lambdas[:8] = np.power(10.0,2-np.arange(8))


#Error arrays
MSE_bs = np.zeros((len(lambdas),max_order))
scores_KFold = np.zeros(k)
MSE_KFOLD = np.zeros((len(lambdas),max_order))

for i in range(1,max_order+1):
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)
    A_curr = A[:,0:currentnot]
    k = 0
    for lasso_par in lambdas: 
        j=0
        RegLasso = linear_model.Lasso(lasso_par, fit_intercept = not rescale)
        for train_index, test_index in rs.split(A_curr):
            A_train = A_curr[train_index]
            z_train = z[train_index]
            A_test = A_curr[test_index]
            z_test = z[test_index]
            ##rescaling
            if rescale == True:
                A_test  -= np.mean(A_train,axis=0)
                A_train -= np.mean(A_train,axis=0)
                z_scale = np.mean(z_train)
                z_train -= z_scale
            ##
            RegLasso.fit(A_train,z_train)
            f_approx_test = RegLasso.predict(A_test)
            scores_KFold[j] = np.sum((f_approx_test - z_test)**2)/np.size(f_approx_test)
            j +=1
        MSE_KFOLD[k][i-1] = np.mean(scores_KFold)
        k += 1
 
min_Kfold_ind = divmod(MSE_KFOLD.argmin(), MSE_KFOLD.shape[1])

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(MSE_KFOLD.T, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'},fmt='.1e')
ax.set_title("Test Accuracy KFOLD 1")
ax.set_ylabel("order")
ax.set_xlabel("log$_{10}(\lambda)$")
ax.set_xticklabels(np.log10(lambdas,out=np.zeros_like(lambdas), where=(lambdas!=0)))
ax.set_yticklabels(range(1,max_order+1))
ax.add_patch(plt.Rectangle((min_Kfold_ind[0], min_Kfold_ind[1]), 1, 1, fc='none', ec='red', lw=2, clip_on=False))    

##Bootstrap
A_train, A_test, z_train, z_test = train_test_split(A, z, test_size=0.2)

for i in range(1,max_order+1):
    k = 0
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)
    ATrCur = A_train[:,0:currentnot]
    ATeCur = A_test[:,0:currentnot]
    for lasso_par in lambdas: 
        ##rescaling
        if rescale == True:
            A_scale = np.mean(ATrCur,axis=0)
            z_scale = np.mean(z_train)
        A_res = ATrCur - A_scale
        ATe_res = ATeCur - A_scale
        z_res = z_train - z_scale
        ##
        RegLasso = linear_model.Lasso(lasso_par, fit_intercept = not rescale)
        RegLasso.fit(A_res,z_res)
        f_approx_test = np.empty((len(z_test), n_bootstrap+1))
        f_approx_test[:,0] = RegLasso.predict(ATe_res)
        for j in range(1,n_bootstrap+1):
            A_res, z_res = resample(ATrCur, z_train, random_state = 1)
            ##rescaling
            if rescale == True:
                A_scale = np.mean(A_res,axis=0)
                z_scale = np.mean(z_res)
            A_res -= A_scale
            ATe_res = ATeCur - A_scale
            z_res -= z_scale
            ##
            f_approx_test[:,j] = RegLasso.predict(ATe_res)
        MSE_bs[k,i-1] = np.mean( np.mean((z_test.reshape(-1, 1) - f_approx_test)**2, axis=1, keepdims=True))
        k += 1

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
print("")
print(f"minimum error KFOLD: {np.min(MSE_KFOLD):.2e}")
print(f"for lambda: {lambdas[min_Kfold_ind[0]]:.1e} and order: {min_Kfold_ind[1]}")

        