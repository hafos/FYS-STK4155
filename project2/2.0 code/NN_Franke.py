#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
@author: simon

This script generates every plot needed for Fitting the Franke Function with the help of a FFNN.
RuntimeWarning: overflow are normal since we get into regions were the NN does not converge!
We did choose the identity as the output activation function everytime.
Here is a quick overview what each function does:

part_a():
    Plots the MSE against different number of layors of hidden layor 
    and neurons (per hidden layor)
    
part_b()
    Plots the MSE against different number of epochs and number of batches

part_c()
    Plots the MSE for sigmoid act. function against different l2 and learningrate values
    
part_c(relu):
    Plots the MSE for relu act. function against different l2 and learningrate values

part_c(leaky_relu()):
    Plots the MSE for leaky relu act. function against different l2 and learningrate values
    
part_d():
    Creates a Plot were for differnt bias initializations the MSE is plottet 
    against the number of epochs

To run a function comment in the call at the bottom of the script
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from FFNN import FFNN
from layer import layer
from FrankeFunction import FrankeFunction
from activation_functions import sigmoid,identity,relu,leaky_relu

class CostOLS:     
    def func(y,ytilde):
        return np.mean(np.power((y-ytilde),2))
    def grad(y,ytilde):
        return (2/ytilde.shape[0])*(ytilde-y)


data, funcval = FrankeFunction(points = 100, sigma=0.25)
X_train, X_test, f_train, f_test = train_test_split(data, funcval, test_size=0.2, random_state=1)

def part_a():
    epochs = 150    
    batches = 32
    eta = 0.1
    l2 = 0.0
    
    layors = np.arange(1,4)
    neurons = np.arange(0,35,5)
    neurons[0] = 1
    
    MSE = np.zeros((len(layors),len(neurons)))

    
    i = 0
    for nr in neurons:
        #One hidden layor
        nn = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = f_train.shape[1], act_fn = identity))
        weights = nn.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches,epochs = epochs, l2 = l2)
        z, a = nn.feed_forward(X_test)
        
        MSE[0,i] = CostOLS.func(f_test,a[-1]) + l2 * np.sum(np.power(weights,2))
        
        #Two hidden layors
        nn = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = f_train.shape[1], act_fn = identity))
        weights = nn.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches,epochs = epochs, l2 = l2)
        z, a = nn.feed_forward(X_test)


        MSE[1,i] = CostOLS.func(f_test,a[-1]) + l2 * np.sum(np.power(weights,2))
        
        #Three hidden layors
        nn = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = f_train.shape[1], act_fn = identity))
        weights = nn.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches,epochs = epochs, l2 = l2)
        z, a = nn.feed_forward(X_test)
        
        MSE[2,i] = CostOLS.func(f_test,a[-1]) + l2 * np.sum(np.power(weights,2))
        
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

def part_b():
    eta = 0.1
    l2 = 0.0
    neurons = 15
    
    batches = [1,4,8,16,32,64,128]
    epochs = np.arange(0,300,50)
    epochs[0] = 5
    
    MSE = np.zeros((len(batches),len(epochs)))
    
    nn = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
    nn.add_layer(layer(neurons = neurons, act_fn = sigmoid))
    nn.add_layer(layer(neurons = f_train.shape[1], act_fn = identity))
    
    i = 0
    for ba in batches:
        j = 0
        for epo in epochs:
            nn.reset()
            weights = nn.train(X_train = X_train, f_train = f_train, 
                               eta=eta, batches = ba ,epochs = epo, l2 = l2)
            z, a = nn.feed_forward(X_test)
            
            MSE[i,j] = CostOLS.func(f_test,a[-1]) + l2 * np.sum(np.power(weights,2))
            
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

def part_c(func = sigmoid):
    epochs = 150
    batches = 32
    neurons = 15
    
    etas = [1e0,1e-1, 1e-2, 1e-3, 1e-4]
    n = 7
    l2s = np.zeros(n)
    l2s[:n-1] = np.power(10.0,1-1*np.arange(n-1))
    
    MSE = np.zeros((len(etas),len(l2s)))
    
    nn = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
    nn.add_layer(layer(neurons = neurons, act_fn = func))
    nn.add_layer(layer(neurons = f_train.shape[1], act_fn = identity))
    
    i = 0
    for eta in etas:
        j = 0
        for l2 in l2s:
            nn.reset()
            weights = nn.train(X_train = X_train, f_train = f_train, 
                               eta=eta, batches = batches ,epochs = epochs, l2 = l2)
            z, a = nn.feed_forward(X_test)
            
            MSE[i,j] = CostOLS.func(f_test,a[-1]) + l2 * np.sum(np.power(weights,2))
            
            j += 1
        i += 1
    
    fig, ax = plt.subplots(figsize=(10, 5))
    MSE[MSE > 10e1] = np.nan
    test = sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='1.3e')
    ax.set_xlabel("lambda")
    ax.set_ylabel("log$_{10}$(eta)")
    ax.set_xticklabels(l2s)
    ax.set_yticklabels(np.log10(etas))
    if type(func) is type:
        ax.set_title(f"NN with {func.__name__}")
    else: 
        ax.set_title(f"NN with {func.__class__.__name__}")
    test.set_facecolor('xkcd:grey')

def part_d():
    batches = 32
    neurons = 15
    eta = 0.01
    l2 = 0
    
    epochs = range(0,20)
    
    MSE = np.zeros((3,len(epochs)))
    
    nn1 = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
    nn1.add_layer(layer(neurons = neurons, act_fn = sigmoid))
    nn1.add_layer(layer(neurons = f_train.shape[1], act_fn = identity))
    
    nn2 = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
    nn2.add_layer(layer(neurons = neurons, act_fn = sigmoid, createbias="ones"))
    nn2.add_layer(layer(neurons = f_train.shape[1], act_fn = identity, createbias="ones"))
    
    nn3 = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
    nn3.add_layer(layer(neurons = neurons, act_fn = sigmoid, createbias="zeros"))
    nn3.add_layer(layer(neurons = f_train.shape[1], act_fn = identity, createbias="zeros"))
    
    i = 0
    for epoch in epochs:
        #random bias
        nn1.reset()
        weights1 = nn1.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches ,epochs = epoch, l2 = l2)
        z1, a1 = nn1.feed_forward(X_test)

        MSE[0,i] = CostOLS.func(f_test,a1[-1]) + l2 * np.sum(np.power(weights1,2))
        
        #bias is 1
        nn2.reset()
        weights2 = nn2.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches ,epochs = epoch, l2 = l2)
        z2, a2 = nn2.feed_forward(X_test)

        MSE[1,i] = CostOLS.func(f_test,a2[-1]) + l2 * np.sum(np.power(weights2,2))
        
        #bias is 0
        nn3.reset()
        weights3 = nn3.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches ,epochs = epoch, l2 = l2)
        z3, a3 = nn3.feed_forward(X_test)

        MSE[2,i] = CostOLS.func(f_test,a3[-1]) + l2 * np.sum(np.power(weights3,2))
        
        i += 1
    
    plt.plot(epochs,MSE[0,:], label = "rand")
    plt.plot(epochs,MSE[1,:], label = "ones")
    plt.plot(epochs,MSE[2,:], label = "zeros")
    plt.xlabel("epochs")
    plt.ylabel("MSE")
    plt.legend()


#part_a()
#part_b()
#part_c()
#part_c(relu)
#part_c(leaky_relu(hyperpar = 0.01))
#part_d()






