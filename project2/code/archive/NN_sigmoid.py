#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import sys

sys.path.append('../src')


from FFNN import FFNN
from gen_data import functions
from sklearn.model_selection import train_test_split
from cost_act_func import activation_functions as act_func
from cost_act_func import CostOLS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dimension = 2
epochs = 150
batches = 128
h_layors = 1
neurons = 25

func = functions(dimension=dimension, sigma=0.25, points = 100)
data, funcval = func.FrankeFunction()
    
learningrates = [1e0,1e-1, 1e-2, 1e-3, 1e-4]
lambda_values = 7
lambdas= np.zeros(lambda_values)
lambdas[:lambda_values-1] = np.power(10.0,1-1*np.arange(lambda_values-1))

MSE = np.zeros((len(learningrates),len(lambdas)))

X_train, X_test, f_train, f_test = train_test_split(data, funcval, test_size=0.2, random_state=1)

split_data = np.array_split(X_train,batches,axis=0)
split_funcval = np.array_split(f_train,batches)

i = 0
for lr in learningrates:
    j = 0
    for param in lambdas:
        costfunc = CostOLS
        nn = FFNN(X_train = X_train, trainval = f_train,
              h_layors = h_layors, h_neurons = neurons, categories = 1,
              CostFunc = costfunc, 
              h_actf = act_func.sigmoid,
              o_actf = act_func.identity,
              methode = "const", learningrate = lr)
        
        np.random.seed(1999) #ensures reproducibility
        for itera in range(epochs):
            for _ in range(batches):
                rd_ind = np.random.randint(batches)
                z,a = nn.FF(split_data[rd_ind])
                nn.backpropagation(z,a,split_funcval[rd_ind],hyperpar=param)
                weights = nn.update_WandB()
        z,a = nn.FF(X_test)
        MSE[i,j] = costfunc.func(f_test,a[len(a)-1]) + param * np.sum(np.power(weights[h_layors],2))
        j += 1
    i += 1

fig, ax = plt.subplots(figsize=(10, 5))
MSE[MSE > 10e1] = np.nan
test = sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='1.3e')
ax.set_xlabel("lambda")
ax.set_ylabel("log$_{10}$(eta)")
ax.set_xticklabels(lambdas)
ax.set_yticklabels(np.log10(learningrates))
ax.set_title("NN with sigmoid")
test.set_facecolor('xkcd:grey')
