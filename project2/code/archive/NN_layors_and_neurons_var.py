#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import sys

sys.path.append('../src')

from FFNN import FFNN
from gen_data import functions
import seaborn as sns
from cost_act_func import activation_functions as act_func
from cost_act_func import CostOLS
import numpy as np
import matplotlib.pyplot as plt


dimension = 2
batches = 1

func = functions(dimension=dimension, sigma=0.25,points= 100)
data, funcval = func.FrankeFunction()

epochs = 150    
batches = 32
layors = np.arange(1,4)
neurons = np.arange(0,30,5)
neurons[0] = 1
costfunc = CostOLS
learningrate = 0.1

MSE = np.zeros((len(layors),len(neurons)))

split_data = np.array_split(data,batches,axis=0)
split_funcval = np.array_split(funcval,batches)

i = 0
for layor in layors:
    j = 0
    for nr in neurons:
        nn = FFNN(X_train = data, trainval = funcval,
                  h_layors = layor, h_neurons = nr, categories = 1,
                  CostFunc = costfunc, 
                  h_actf = act_func.sigmoid,
                  o_actf = act_func.identity,
                  methode = "const", learningrate = learningrate)
    
        for itera in range(epochs):
            for _ in range(batches):
                rd_ind = np.random.randint(batches)
                z,a = nn.FF(split_data[rd_ind])
                nn.backpropagation(z,a,split_funcval[rd_ind])
                nn.update_WandB()
        z,a = nn.FF()
        MSE[i,j] = CostOLS.func(funcval,a[len(a)-1])
        j += 1
    i += 1

fig, ax = plt.subplots(figsize=(10, 5))
MSE[MSE > 10e1] = np.nan
heatmap = sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='1.3e')
ax.set_xlabel("neurons")
ax.set_ylabel("hidden layors")
ax.set_xticklabels(neurons)
ax.set_yticklabels(layors)
ax.set_title("NN with sigmoid")
heatmap.set_facecolor('xkcd:grey')