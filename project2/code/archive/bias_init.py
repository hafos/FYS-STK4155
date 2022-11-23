#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import sys

sys.path.append('../src')


from FFNN import FFNN
from gen_data import functions
from cost_act_func import activation_functions as act_func
from cost_act_func import CostOLS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

dimension = 2
epochs = range(0,100)
batches = 32

func = functions(dimension=dimension, sigma=0.25, points= 100)
data, funcval = func.FrankeFunction()

learningrate = 0.01
neurons = 5
hlayors = 1

MSE = np.zeros((3,len(epochs)))

split_data = np.array_split(data,batches,axis=0)
split_funcval = np.array_split(funcval,batches)


costfunc = CostOLS
nn_zeros = FFNN(X_train = data, trainval = funcval,
      h_layors = hlayors, h_neurons = neurons, categories = 1,
      createbias = "zeros", CostFunc = costfunc, 
      h_actf = act_func.sigmoid,
      o_actf = act_func.identity,
      methode = "const", learningrate = learningrate)

nn_ones = FFNN(X_train = data, trainval = funcval,
      h_layors = hlayors, h_neurons = neurons, categories = 1,
      createbias = "ones", CostFunc = costfunc, 
      h_actf = act_func.sigmoid,
      o_actf = act_func.identity,
      methode = "const", learningrate = learningrate)


nn_rand = FFNN(X_train = data, trainval = funcval,
      h_layors = hlayors, h_neurons = neurons, categories = 1,
      CostFunc = costfunc, 
      h_actf = act_func.sigmoid,
      o_actf = act_func.identity,
      methode = "const", learningrate = learningrate)

i = 0
for epoch in epochs:
    np.random.seed(1999) #ensures reproducibility
    for itera in range(epoch):
        for _ in range(batches):
            rd_ind = np.random.randint(batches)
            z_zeros,a_zeros = nn_zeros.FF(split_data[rd_ind]) 
            z_ones,a_ones = nn_ones.FF(split_data[rd_ind]) 
            z_rand,a_rand = nn_rand.FF(split_data[rd_ind]) 
            
            nn_zeros.backpropagation(z_zeros,a_zeros,split_funcval[rd_ind])
            nn_ones.backpropagation(z_ones,a_ones,split_funcval[rd_ind])
            nn_rand.backpropagation(z_rand,a_rand,split_funcval[rd_ind])
            
            nn_zeros.update_WandB()
            nn_ones.update_WandB()
            nn_rand.update_WandB()
            
    z_zeros,a_zeros = nn_zeros.FF()
    z_ones,a_ones = nn_ones.FF()
    z_rand,a_rand = nn_rand.FF()
    
    MSE[0,i] = costfunc.func(funcval,a_zeros[len(a_zeros)-1])
    MSE[1,i] = costfunc.func(funcval,a_ones[len(a_ones)-1])
    MSE[2,i] = costfunc.func(funcval,a_rand[len(a_rand)-1])
    i += 1

plt.plot(epochs,MSE[0,:], label = "zeros")
plt.plot(epochs,MSE[1,:], label = "ones")
plt.plot(epochs,MSE[2,:], label = "rand")
plt.legend()
