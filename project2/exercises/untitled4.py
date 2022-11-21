#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""


from FFNN import FFNN
from gen_data import functions
from cost_act_func import activation_functions as act_func
from cost_act_func import Ridge
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append('../classes')

dimension = 2
epochs = 500
batches = 1

func = functions(dimension=dimension, sigma=0.0,points= 100)
data, funcval = func.FrankeFunction()
    
learningrates = [1e-1, 1e-2, 1e-3, 1e-4]
lambdas = [0]
#lambdas= np.zeros(5)
#lambdas[:4] = np.power(10.0,-1+-1*np.arange(4))

MSE = np.zeros((len(learningrates),len(lambdas)))

i = 0
for lr in learningrates:
    j = 0
    for param in lambdas:
        costfunc = Ridge(hyperpar=param)
        nn = FFNN(X_train = data, trainval = funcval,
              h_layors = 1, h_neurons = 30, categories = 1,
              CostFunc = costfunc, 
              h_actf = act_func.sigmoid,
              o_actf = act_func.identity,
              methode = "const", learningrate = lr)
        for itera in range(epochs):
            for _ in range(batches):
                z,a = nn.FF()
                nn.backpropagation(z,a)
                nn.update_WandB()
        z,a = nn.FF()
        MSE[i,j] = costfunc.func(funcval,a[len(a)-1])
        j += 1
    i += 1

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='1.3e')
ax.set_xlabel("lambda")
ax.set_ylabel("log$_{10}$(eta)")
ax.set_xticklabels(lambdas)
ax.set_yticklabels(np.log10(learningrates))
