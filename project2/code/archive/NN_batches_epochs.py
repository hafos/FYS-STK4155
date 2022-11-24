#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

from sklearn.model_selection import train_test_split
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

func = functions(dimension=dimension, sigma=0.25,points= 100)
data, funcval = func.FrankeFunction()

X_train, X_test, f_train, f_test = train_test_split(data, funcval, test_size=0.2, random_state=1)

 

h_layors = 1
neurons = 25
learningrate = 0.1
batches = [1,4,8,16,32,64,128]
epochs = np.arange(0,300,50)
epochs[0] = 5
costfunc = CostOLS

MSE = np.zeros((len(batches),len(epochs)))

i = 0
for batch in batches:
    j = 0
    split_data = np.array_split(X_train,batch,axis=0)
    split_funcval = np.array_split(f_train,batch)
    for epoch in epochs:
        nn = FFNN(X_train = X_train, trainval = f_train,
                  h_layors = h_layors, h_neurons = neurons, categories = 1,
                  CostFunc = costfunc, 
                  h_actf = act_func.sigmoid,
                  o_actf = act_func.identity,
                  methode = "const", learningrate = learningrate)
    
        np.random.seed(1999) #ensures reproducibility
        for itera in range(epoch):
            for _ in range(batch):
                rd_ind = np.random.randint(batch)
                z,a = nn.FF(split_data[rd_ind])
                nn.backpropagation(z,a,split_funcval[rd_ind])
                nn.update_WandB()
        z,a = nn.FF(X_test)
        MSE[i,j] = CostOLS.func(f_test,a[len(a)-1])
        j += 1
    i += 1

fig, ax = plt.subplots(figsize=(10, 5))
MSE[MSE > 10e1] = np.nan
heatmap = sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='1.3e')
ax.set_xlabel("epochs")
ax.set_ylabel("batches")
ax.set_xticklabels(epochs)
ax.set_yticklabels(batches)
ax.set_title("NN with sigmoid")
heatmap.set_facecolor('xkcd:grey')