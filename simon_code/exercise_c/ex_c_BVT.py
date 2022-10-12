#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 16:21:46 2022

@author: simon
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#Import all needed functions from different py file
from funcs import functions as fnc

##global Parameters

#IF sigma is zero there is no noise in the Model
sigma = 0.1
#Number of x & y points, total amount of datapoints is this squared
points = 30
#Programm will fit a polnyomial up to this order
max_order = 18
#number of bootstraps
n_bootstrap = 50

##create data and designe matrix 

x,y,fval = fnc.makedata(points,sigma,fnc.FrankeFunction)
variables=[x,y]
A = fnc.DM(variables,max_order)
fnc.rescale = False
#split data into test & train
A_train, A_test, f_train, f_test = train_test_split(A, fval, test_size=0.2, random_state = 1)

##create all needed arrays 

MSE_train = np.zeros(max_order)
MSE_test = np.zeros(max_order)
BIAS = np.zeros(max_order)
var = np.zeros(max_order)
ftr_aval = np.zeros(len(f_train))

for i in range(1,max_order+1):
    #current number of tearms also via complete homogeneous symmetric polynomials
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)
    #select only the terms of the full desinge matrix needed for the current order
    ATrCur = A_train[:,0:currentnot]
    ATeCur = A_test[:,0:currentnot]
    #for fig 2.11 of Hastie, Tibshirani, and Friedman
    beta = fnc.OLS(ATrCur,f_train)
    ftr_aval = ATrCur @ beta
    #Calcuate the errors
    MSE_test[i-1], fte_aval = fnc.bootstrap(n_bootstrap,ATrCur,ATeCur,f_train,f_test,fnc.OLS)
    MSE_train[i-1] = np.mean(np.power(f_train-ftr_aval,2))
    BIAS[i-1] = np.mean( (f_test.reshape(-1, 1) - np.mean(fte_aval, axis=1, keepdims=True))**2 )
    var[i-1] = np.mean( np.var(fte_aval, axis=1, keepdims=True) )
    print("subtrac",MSE_test[i-1]-(BIAS[i-1]+var[i-1]))

##reproduce fig 2.11 of Hastie, Tibshirani, and Friedman
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_train, label='train', color='orange', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_test, label='test', color='blue', s=15)
plt.legend(loc = "upper center")
plt.show()

##Bias Variance tradeoff
plt.scatter(np.arange(1, max_order+1, 1.0), BIAS, label='BIAS', color='orange', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_test, label='test', color='blue', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), var, label='var', color='red', s=15)     
plt.legend()
plt.show()
    


