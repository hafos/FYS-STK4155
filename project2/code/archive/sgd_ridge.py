#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:29:29 2022

@author: simon
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:33:41 2022

@author: simon
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import seaborn as sns
import sys


sys.path.append('../classes')

from S_Grad_Decent import StochGradDecent
from gen_data import functions
from cost_act_func import Ridge_beta
from Linear_Regression import LinearRegression


dimension = 2
order = 6

epochs = 150
batches = 32

func = functions(order = order, dimension=dimension, sigma=0.0, points= 100)

data, funcval = func.FrankeFunction()
poly = PolynomialFeatures(degree=order)
X = poly.fit_transform(data)

reg = LinearRegression(X,funcval)
 
learningrates = [1e-1, 1e-2, 1e-3, 1e-4]
lambdas= np.zeros(5)
lambdas[:4] = np.power(10.0,-1+-1*np.arange(4))

MSE = np.zeros((len(learningrates),len(lambdas)))

i = 0
for lr in learningrates:
    j = 0
    for params in lambdas:
        costfunc = Ridge_beta(hyperpar = params)
        sd = StochGradDecent(X,funcval,costfunc=costfunc)
        beta = sd.const(epochs = epochs, batches = batches, learningrate = lr)
        MSE[i,j] = costfunc.func(funcval,X,beta)
        j += 1
    i += 1

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='1.3e')
ax.set_xlabel("lambda")
ax.set_ylabel("log$_{10}$(eta)")
ax.set_xticklabels(lambdas)
ax.set_yticklabels(np.log10(learningrates))



