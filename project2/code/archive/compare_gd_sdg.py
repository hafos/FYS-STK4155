#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import sys

sys.path.append('../src')

from Grad_Decent import GradDecent
from S_Grad_Decent import StochGradDecent
from gen_data import functions
from cost_act_func import CostOLS_beta


func = functions(dimension=2, sigma=0.25 ,points= 100)
costfunc = CostOLS_beta


data, funcval = func.FrankeFunction()

order = 6
poly = PolynomialFeatures(degree=order)
X = poly.fit_transform(data)


gd = GradDecent(X,funcval,costfunc=costfunc)
sd = StochGradDecent(X,funcval,costfunc=costfunc)


print ("Different learningrates for constant batchsize: ...", end = "")
learningrates = [10e-2, 10e-3]
epochs = np.arange(200)

MSE_gd = np.zeros((len(learningrates),len(epochs)))
MSE_sd = np.zeros((len(learningrates),len(epochs)))

i = 0
for lr in learningrates:
    j = 0
    for it in epochs:
        beta_gd = gd.const(iterations = it, learningrate = lr)
        beta_sd = sd.const(epochs = it, batches = 4, learningrate = lr)
        MSE_gd[i,j] = costfunc.func(funcval,X,beta_gd)
        MSE_sd[i,j] = costfunc.func(funcval,X,beta_sd)
        j +=1
    i+=1
    
colors = ["green", "blue"]
for i in range(len(learningrates)):
    plt.plot(epochs, MSE_gd[i,:], label = f"GD with eta = {learningrates[i]}",
             color = colors[i])
    plt.plot(epochs, MSE_sd[i,:], label = f"SGD with eta = {learningrates[i]}",
             linestyle = "dashed", color = colors[i])
    plt.legend()

plt.ylabel("MSE")
plt.xlabel("epochs")
plt.show()

print("[DONE]")


print ("Different batchsizes for constant learningrate: ...", end = "")
epochs = np.arange(0,200)
batchsizes = [1,4,16,64]

MSE_gd = np.zeros(len(epochs))
MSE_sd = np.zeros((len(batchsizes),len(epochs)))

i = 0
for it in epochs:
    j = 0
    for bs in batchsizes:
        beta_sd = sd.const(epochs = it, batches = bs, learningrate = 0.01)
        MSE_sd[j,i] = costfunc.func(funcval,X,beta_sd)
        j +=1
    i += 1

for j in range(MSE_sd.shape[0]):
    plt.plot(epochs,MSE_sd[j,:],label = f"Number of batches = {batchsizes[j]}")
plt.legend()
plt.ylabel("MSE")
plt.xlabel("epochs")
plt.show
print("[DONE]")


