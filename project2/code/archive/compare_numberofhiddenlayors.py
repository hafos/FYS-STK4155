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

func = functions(dimension=dimension, sigma=0.0,points= 100)
data, funcval = func.FrankeFunction()

costfunc = CostOLS


neurons = np.arange(2,30)
MSE = np.zeros((2,len(neurons)))

i = 0
for nr in neurons:
    nn1 = FFNN(X_train = data, trainval = funcval,
              h_layors = 1, h_neurons = nr, categories = 1,
              CostFunc = costfunc, 
              h_actf = act_func.sigmoid,
              o_actf = act_func.identity,
              methode = "const", learningrate = 0.1)
    nn2 = FFNN(X_train = data, trainval = funcval,
              h_layors = 2, h_neurons = nr, categories = 1,
              CostFunc = costfunc, 
              h_actf = act_func.sigmoid,
              o_actf = act_func.identity,
              methode = "const", learningrate = 0.1)
    epochs = 100
    for itera in range(epochs):
        for _ in range(batches):
            z1,a1 = nn1.FF()
            z2,a2 = nn2.FF()
            nn1.backpropagation(z1,a1)
            nn1.update_WandB()
            nn2.backpropagation(z2,a2)
            nn2.update_WandB()
    z1,a1 = nn1.FF()
    z2,a2 = nn2.FF()
    MSE[0,i] = CostOLS.func(funcval,a1[len(a1)-1])
    MSE[1,i] = CostOLS.func(funcval,a2[len(a2)-1])
    i += 1

plt.scatter(neurons, MSE[0,:])
plt.scatter(neurons, MSE[1,:])