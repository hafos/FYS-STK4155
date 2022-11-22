#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:54:29 2022

@author: simon
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import sys

sys.path.append('../src')

from S_Grad_Decent import StochGradDecent
from gen_data import functions
from cost_act_func import CostOLS_beta


func = functions(dimension=2, sigma=0.25 ,points= 100)
costfunc = CostOLS_beta


data, funcval = func.FrankeFunction()

order = 7
poly = PolynomialFeatures(degree=order)
X = poly.fit_transform(data)

sd = StochGradDecent(X,funcval,costfunc=costfunc)
epochs = 100
eta = 1e-1
batches = 512

mompar = [0.6,0.5,0.4,0.3,0.2,0.1,0]
MSE = np.zeros(len(mompar))

i = 0
for par in mompar:
    beta_momentum = sd.momentum(epochs = epochs, batches = batches, learningrate = eta,
                            delta_momentum = par)
    MSE[i] = costfunc.func(funcval,X,beta_momentum)
    i += 1

plt.scatter(mompar,MSE)
plt.title("Momentum based stochastic gradient decent")
plt.xlabel("momentum parameter")
plt.ylabel("MSE")
plt.show()

learningrates = [0.001,0.01,0.1,1]
MSE_ada = np.zeros(len(learningrates))
MSE_adam = np.zeros(len(learningrates))
MSE_rms = np.zeros(len(learningrates))

i = 0
for lr in learningrates:
    beta_ada = sd.adagrad(epochs = epochs, batches = batches, learningrate = lr)
    beta_rms = sd.rmsprop(epochs = epochs, batches = batches, learningrate = lr)
    beta_adam= sd.adam(epochs = epochs, batches = batches, learningrate = lr)
    MSE_ada[i] = costfunc.func(funcval,X,beta_ada)
    MSE_rms[i] = costfunc.func(funcval,X,beta_rms)
    MSE_adam[i] = costfunc.func(funcval,X,beta_adam)
    i += 1

print(f"learningrates: \t {learningrates}")
print(f"MSE adagrad: \t {MSE_ada}")
print(f"MSE rmsprop: \t {MSE_rms}")
print(f"MSE adam: \t {MSE_adam}")


#beta_adamom = sd.adagrad(momentum = True,learningrate = 1)
#beta_rmsprop = sd.rmsprop()
#beta_adam = sd.adam()
