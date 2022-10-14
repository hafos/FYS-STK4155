#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 15:32:12 2022

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
max_order = 20
##create data and designe matrix 

x,y,fval = fnc.makedata(points,sigma,fnc.FrankeFunction)
fnc.rescale = False
variables=[x,y]
A = fnc.DM(variables,max_order)

#Folds for crossvalidation
kfold = [2,5]

#Error arrays
MSE_Kfold = np.zeros((len(kfold),max_order))


for i in range(1,max_order+1):
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)
    A_curr = A[:,0:currentnot] + 0
    k = 0
    for folds in kfold: 
        MSE_Kfold[k,i-1] = fnc.crossval(folds,A_curr,fval,fnc.OLS)
        k += 1

for i in range(0,len(kfold)):       
    plt.scatter(np.arange(1, max_order+1, 1.0), MSE_Kfold[i], label=str(kfold[i]), s=15)    
plt.legend()
plt.show()

