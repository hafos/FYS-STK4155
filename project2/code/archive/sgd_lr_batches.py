#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:16:47 2022

@author: simon
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import seaborn as sns
import sys

sys.path.append('../src')

from S_Grad_Decent import StochGradDecent
from gen_data import functions
from cost_act_func import CostOLS_beta


func = functions(dimension=2, sigma=0.25 ,points= 100)
costfunc = CostOLS_beta


data, funcval = func.FrankeFunction()

order = 6
poly = PolynomialFeatures(degree=order)
X = poly.fit_transform(data)

sd = StochGradDecent(X,funcval,costfunc=costfunc)
epochs = 100
learningrates = [1e-1, 1e-2, 1e-3, 1e-4]
batchsizes = [1,2,4,8,16,32,64,128,256,512,1024]

MSE = np.zeros((len(batchsizes),len(learningrates)))

i = 0
for bs in batchsizes:
    j = 0
    for lr in learningrates:
        beta = sd.const(epochs = epochs, batches = bs, learningrate = lr)
        MSE[i,j] = costfunc.func(funcval,X,beta)
        j += 1
    i += 1

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='1.3e')
ax.set_xlabel("log$_{10}$(eta)")
ax.set_ylabel("batchsize")
ax.set_xticklabels(np.log10(learningrates))
ax.set_yticklabels(batchsizes)

plt.show() 