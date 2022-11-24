#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split 
from sklearn.datasets import load_breast_cancer
import seaborn as sns
import time

import sys

sys.path.append('../src')

from FFNN import FFNN
from cost_act_func import activation_functions as act_func
from cost_act_func import cross_entropy

# Load the data
cancer = load_breast_cancer()
data = cancer.data
target = cancer.target.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(data,target,random_state=0)

epochs = 150
batches = 32

split_data = np.array_split(X_train,batches,axis=0)
split_funcval = np.array_split(y_train,batches)

costfunc = cross_entropy
nn = FFNN(X_train = X_train, trainval = y_train,
      h_layors = 1, h_neurons = 25, categories = 1,
      CostFunc = costfunc,
      h_actf = act_func.sigmoid,
      o_actf = act_func.sigmoid,
      methode = "const", learningrate = 0.1)

np.random.seed(1999) #ensures reproducibility
for itera in range(epochs):
    for _ in range(batches):
        rd_ind = np.random.randint(batches)
        z,a = nn.FF(split_data[rd_ind])
        nn.backpropagation(z,a,split_funcval[rd_ind])
        nn.update_WandB()
z,a = nn.FF(X_test)
z_train,a_train = nn.FF(X_train)
MSE = costfunc.func(y_test,a[len(a)-1])
print(costfunc.func(y_train,a_train[len(a)-1]))

print(MSE)
