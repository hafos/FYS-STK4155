#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import sys

sys.path.append('../classes')


from FFNN import FFNN
from gen_data import functions
from cost_act_func import activation_functions as act_func
from cost_act_func import CostOLS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

dimension = 2
epochs = 100
batches = 300
learningrate = 0.1

func = functions(dimension=dimension, sigma=0.25, points= 100)
data, funcval = func.FrankeFunction()

learningrates = [1e-2, 1e-3, 1e-4]
MSE = np.zeros((len(learningrates),1))

split_data = np.array_split(data,batches,axis=0)
split_funcval = np.array_split(funcval,batches)

i = 0
for lr in learningrates:
    costfunc = CostOLS
    nn = FFNN(X_train = data, trainval = funcval,
          h_layors = 1, h_neurons = 30, categories = 1,
          CostFunc = costfunc, 
          h_actf = act_func.relu,
          o_actf = act_func.identity,
          methode = "const", learningrate = lr)
        
    np.random.seed(1999) #ensures reproducibility
    for itera in range(epochs):
        for _ in range(batches):
            rd_ind = np.random.randint(batches)
            z,a = nn.FF(split_data[rd_ind])
            print(a[len(a)-1])   
            nn.backpropagation(z,a,split_funcval[rd_ind])
            nn.update_WandB()
    z,a = nn.FF()
    MSE[i] = costfunc.func(funcval,a[len(a)-1])
    i += 1

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='1.3e')
ax.set_xlabel("lambda")
ax.set_ylabel("log$_{10}$(eta)")
ax.set_xticklabels([1])
ax.set_yticklabels(np.log10(learningrates))
