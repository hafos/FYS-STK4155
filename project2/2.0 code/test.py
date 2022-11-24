#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:58:18 2022

@author: simon
"""

import numpy as np
from FFNN import FFNN
from layer import layer
from FrankeFunction import FrankeFunction
from sklearn.model_selection import train_test_split
from notyet import CostOLS
from notyet import activation_functions

data, funcval = FrankeFunction()
sigmoid = activation_functions.sigmoid
identity = activation_functions.identity

l2 = 0

X_train, X_test, f_train, f_test = train_test_split(data, funcval, test_size=0.2, random_state=1)

nn = FFNN(parameters=X_train.shape[1], cost_func = CostOLS)
nn.add_layer(layer(neurons = 25, act_fn = sigmoid))
nn.add_layer(layer(neurons = f_train.shape[1], act_fn = identity))
weights = nn.train(X_train = X_train, f_train = f_train)
z, a = nn.feed_forward(X_test)
MSE = CostOLS.func(f_test,a[-1]) + l2 * np.sum(np.power(weights,2))
