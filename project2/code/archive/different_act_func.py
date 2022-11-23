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
epochs = range(5,50)
batches = 32

func = functions(dimension=dimension, sigma=0.25, points= 100)
data, funcval = func.FrankeFunction()

learningrate = 0.01
neurons = 20
hlayors = 2

MSE = np.zeros((3,len(epochs)))

split_data = np.array_split(data,batches,axis=0)
split_funcval = np.array_split(funcval,batches)


costfunc = CostOLS
nn_sig = FFNN(X_train = data, trainval = funcval,
      h_layors = hlayors, h_neurons = neurons, categories = 1,
      CostFunc = costfunc, 
      h_actf = act_func.sigmoid,
      o_actf = act_func.identity,
      methode = "const", learningrate = learningrate)

nn_relu = FFNN(X_train = data, trainval = funcval,
      h_layors = hlayors, h_neurons = neurons, categories = 1,
      CostFunc = costfunc, 
      h_actf = act_func.relu,
      o_actf = act_func.identity,
      methode = "const", learningrate = learningrate)


nn_lrelu = FFNN(X_train = data, trainval = funcval,
      h_layors = hlayors, h_neurons = neurons, categories = 1,
      CostFunc = costfunc, 
      h_actf = act_func.leaky_relu(hyperpar = 0.01),
      o_actf = act_func.identity,
      methode = "const", learningrate = learningrate)

i = 0
for epoch in epochs:
    np.random.seed(1999) #ensures reproducibility
    for itera in range(epoch):
        for _ in range(batches):
            rd_ind = np.random.randint(batches)
            z_sig,a_sig = nn_sig.FF(split_data[rd_ind]) 
            z_relu,a_relu = nn_relu.FF(split_data[rd_ind]) 
            z_lrelu,a_lrelu = nn_lrelu.FF(split_data[rd_ind]) 
            
            nn_sig.backpropagation(z_sig,a_sig,split_funcval[rd_ind])
            nn_relu.backpropagation(z_relu,a_relu,split_funcval[rd_ind])
            nn_lrelu.backpropagation(z_lrelu,a_lrelu,split_funcval[rd_ind])
            
            nn_sig.update_WandB()
            nn_relu.update_WandB()
            nn_lrelu.update_WandB()
            
    z_sig,a_sig = nn_sig.FF()
    z_relu,a_relu = nn_relu.FF()
    z_lrelu,a_lrelu = nn_lrelu.FF()
    
    MSE[0,i] = costfunc.func(funcval,a_sig[len(a_sig)-1])
    MSE[1,i] = costfunc.func(funcval,a_relu[len(a_relu)-1])
    MSE[2,i] = costfunc.func(funcval,a_lrelu[len(a_lrelu)-1])
    i += 1

plt.plot(epochs,MSE[0,:], label = "sig")
plt.plot(epochs,MSE[1,:], label = "relu")
plt.plot(epochs,MSE[2,:], label = "lrelu")
plt.legend()
