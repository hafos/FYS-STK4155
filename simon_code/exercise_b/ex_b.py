#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:38:10 2022

@author: simon
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from funcs import functions as fnc
from sklearn.model_selection import train_test_split

##global Parameters

#IF sigma is zero there is no noise in the Model
sigma = 0.1
#Number of x & y points, total amount of datapoints is this squared
points = 40
#Programm will fit a polnyomial up to this order
max_order = 5

##create data and designe matrix 

x,y,fval = fnc.makedata(points,sigma,fnc.FrankeFunction)
variables=[x,y]
A = fnc.DM(variables,max_order)

##create all needed arrays

#the dimension of the array needed is given complete homogeneous symmetric polynomial
numbofterms = sp.special.comb(len(variables) + max_order,max_order,exact=True)
beta = np.full((max_order, numbofterms), np.nan)
MSE_test = np.zeros(max_order)
MSE_train = np.zeros(max_order)
R2_test = np.zeros(max_order)
R2_train = np.zeros(max_order)

#split data into test & train
A_train, A_test, f_train, f_test = train_test_split(A, fval, test_size=0.2, random_state = 1)

for i in range(1,max_order+1):
    #current number of tearms also via complete homogeneous symmetric polynomials
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)
    #select only the terms of the full desinge matrix needed for the current order
    ATrCur = A_train[:,0:currentnot] + 0 
    ATeCur = A_test[:,0:currentnot] + 0
    #calc both errors and store the betas in the process
    beta[i-1][:currentnot] = fnc.OLS(ATrCur,f_train)
    fte_aval = ATeCur @ beta[i-1][~np.isnan(beta[i-1])]
    ftr_aval = ATrCur @ beta[i-1][~np.isnan(beta[i-1])]
    MSE_test[i-1] = np.mean(np.power(f_test-fte_aval,2))
    MSE_train[i-1] = np.mean(np.power(f_train-ftr_aval,2))
    R2_test[i-1] = fnc.R2(f_test,fte_aval)
    R2_train[i-1] = fnc.R2(f_train,ftr_aval)

plt.scatter(np.arange(1, max_order+1, 1.0), MSE_test, label='test', color='orange', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_train, label='train', color='red', s=15)
plt.legend()
plt.show()

plt.scatter(np.arange(1, max_order+1, 1.0), R2_test, label='test', color='blue', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), R2_train, label='train', color='green', s=15)
plt.legend()
plt.show()