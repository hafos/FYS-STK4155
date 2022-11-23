#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

from FFNN import FFNN
from gen_data import functions
from cost_act_func import activation_functions as act_func
from cost_act_func import CostOLS
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../src')

dimension = 2
batches = 1

func = functions(dimension=dimension, sigma=0.25,points= 100)
data, funcval = func.FrankeFunction()

costfunc = CostOLS

epochs = 150    
batches = 32
layors = np.arange(1,10)
neurons = 20
MSE = np.zeros(len(layors))

split_data = np.array_split(data,batches,axis=0)
split_funcval = np.array_split(funcval,batches)

i = 0
for layor in layors:
    nn = FFNN(X_train = data, trainval = funcval,
              h_layors = layor, h_neurons = neurons, categories = 1,
              CostFunc = costfunc, 
              h_actf = act_func.sigmoid,
              o_actf = act_func.identity,
              methode = "const", learningrate = 0.1)

    for itera in range(epochs):
        for _ in range(batches):
            rd_ind = np.random.randint(batches)
            z,a = nn.FF(split_data[rd_ind])
            nn.backpropagation(z,a,split_funcval[rd_ind])
            nn.update_WandB()
    z,a = nn.FF()
    MSE[i] = CostOLS.func(funcval,a[len(a)-1])
    i += 1

plt.plot(layors, MSE)