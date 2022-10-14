#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 02:25:56 2022

@author: simon
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns

from sklearn.model_selection import train_test_split

#Import all needed functions from different py file
from funcs import functions as fnc


##global Parameters

#IF sigma is zero there is no noise in the Model
sigma = 0.1
#Number of x & y points, total amount of datapoints is this squared
points = 8
#Programm will fit a polnyomial between min and max order
min_order = 5
max_order = 15
#number of bootstraps
n_bootstrap = 10

#different number of data points
datapoints = np.arange(20,40,5)


#Error arrays
MSE_nop = np.zeros((len(datapoints),max_order-min_order+1))
fnc.rescale = False

k=0
for points in datapoints:
    ##create data and designe matrix
    x,y,fval = fnc.makedata(points,sigma,fnc.FrankeFunction)
    variables=[x,y]
    A = fnc.DM(variables,max_order)
    #split data into test & train
    A_train, A_test, f_train, f_test = train_test_split(A, fval, test_size=0.2, random_state = 1)
    i = 0
    for order in range(min_order,max_order+1):
        #current number of tearms also via complete homogeneous symmetric polynomials
        currentnot = sp.special.comb(len(variables) + i,i,exact=True)
        #select only the terms of the full desinge matrix needed for the current order
        ATrCur = A_train[:,0:currentnot] + 0
        ATeCur = A_test[:,0:currentnot] + 0
        MSE_nop[k,i] = fnc.bootstrap(n_bootstrap,ATrCur,ATeCur,f_train,f_test,fnc.OLS)[0]
        i += 1
    k += 1

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(MSE_nop.T, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'},fmt='.1e')
ax.set_title("Different number of datapoints")
ax.set_ylabel("order")
ax.set_xlabel("datapoints")
ax.set_xticklabels(np.power(datapoints,2))
ax.set_yticklabels(np.arange(min_order,max_order+1,1))