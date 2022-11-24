#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
@author: simon

To run a function comment in the call at the bottom of the script
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from FFNN import FFNN
from layer import layer
from FrankeFunction import FrankeFunction
from activation_functions import sigmoid,identity,relu,leaky_relu
 
class cross_entropy:
    """"
    Defines the Cost function and its derivative used for all calculations in this script
    """
    def func(y,ytilde):
        funcval = np.sum(y*np.log(ytilde) + (1-y)*np.log(1-ytilde))
        return -1/y.size * np.sum(funcval)
    def grad(y,ytilde):
        return -1/y.size*(y/(ytilde)-(1-y)/(1-ytilde))

cancer = load_breast_cancer()
data = cancer.data
target = cancer.target.reshape(-1,1)

X_train, X_test, f_train, f_test = train_test_split(data,target,random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def part_a():
    epochs = 200
    batches = 32
    neurons = 20
    eta = 0.01
    l2 = 0
    
    nn = FFNN(parameters=X_train.shape[1], cost_fn = cross_entropy)
    nn.add_layer(layer(neurons = neurons, act_fn = sigmoid))
    nn.add_layer(layer(neurons = f_train.shape[1], act_fn = sigmoid))
    weights = nn.train(X_train = X_train, f_train = f_train, 
                       eta=eta, batches = batches,epochs = epochs, l2 = l2)
    z, a = nn.feed_forward(X_test)
    
    MSE = cross_entropy.func(f_test,a[-1])
    prediction = a[-1].round()
    accuracy = np.sum(prediction == f_test)/f_test.shape[0]
    print(accuracy)
    print(MSE)

part_a()

