#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 15:17:12 2022

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
max_order = 25

##create data and designe matrix 

x,y,fval = fnc.makedata(points,sigma,fnc.FrankeFunction)
fnc.rescale = False
variables=[x,y]
A = fnc.DM(variables,max_order)

##Folds for crossvalidation
kfold = 4

#Error arrays

MSE_Kfold = np.zeros(max_order)
MSE_bs = np.zeros(max_order)



for i in range(1,max_order+1):
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)
    A_curr = A[:,0:currentnot] + 0
    MSE_Kfold[i-1] = fnc.crossval(kfold,A_curr,fval,fnc.OLS)
    
##Bootstrap
#number of Folds
n_bootstrap = 4

#split data into test & train
A_train, A_test, f_train, f_test = train_test_split(A, fval, test_size=0.2, random_state = 1)
for i in range(1,max_order+1):
    #current number of tearms also via complete homogeneous symmetric polynomials
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)
    #select only the terms of the full desinge matrix needed for the current order
    ATrCur = A_train[:,0:currentnot] + 0
    ATeCur = A_test[:,0:currentnot] + 0
    MSE_bs[i-1] = fnc.bootstrap(n_bootstrap,ATrCur,ATeCur,f_train,f_test,fnc.OLS)[0]
    
#Bootstrap
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_bs, label='MSE_bs', color='orange', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_Kfold, label='MSE_KFOLD', color='blue', s=15)    
plt.legend()
plt.show()
