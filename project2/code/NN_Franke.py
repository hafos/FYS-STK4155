#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
@author: simon

This script generates every plot needed for fitting the Franke Function with the help of a FFNN.
RuntimeWarning: overflow are normal since we get into regions were the NN does not converge!
We did choose the identity as the output activation function everytime.
Here is a quick overview what each function does:

plot_neurons_vs_layers(batches, epochs, eta)
    Plots the MSE against different number of layers of hidden layer 
    and neurons (per hidden layer)
    
plot_epochs_vs_batches(neurons, h_layers, eta)
    Plots the MSE against different number of epochs and number of operations

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
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split

from FFNN import FFNN
from layer import layer
from FrankeFunction import FrankeFunction
from activation_functions import sigmoid,identity,relu,leaky_relu,tanh

# Save figures (Y/N)
save = "Y"

class CostOLS:     
    """"
    Defines the Cost function and its derivative used for all calculations in this script
    """
    def func(y,ytilde):
        return np.mean(np.power((y-ytilde),2))
    def grad(y,ytilde):
        return (2/ytilde.shape[0])*(ytilde-y)

# Generate data and split it into test/train sets
data, funcval = FrankeFunction(points = 100, sigma=0.25)
X_train, X_test, f_train, f_test = train_test_split(data, funcval, test_size=0.2, random_state=1)

def plot_neurons_vs_layers(batches=32, epochs=150, eta=0.1, l2=0.0):
    print("MSE for different number of hidden layers and neurons:...")
    
    layors = np.arange(1, 4)
    neurons = np.arange(0, 35, 5)
    neurons[0] = 1
    
    MSE = np.zeros((len(layors),len(neurons)))

    i = 0
    for nr in neurons:
        #One hidden layor
        nn = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = f_train.shape[1], act_fn = identity))
        weights = nn.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches, epochs = epochs, l2 = l2)
        z, a = nn.feed_forward(X_test)
        
        MSE[0,i] = CostOLS.func(f_test,a[-1]) + l2 * np.sum(np.power(weights,2))
        
        #Two hidden layors
        nn = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = f_train.shape[1], act_fn = identity))
        weights = nn.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches, epochs = epochs, l2 = l2)
        z, a = nn.feed_forward(X_test)

        MSE[1,i] = CostOLS.func(f_test,a[-1]) + l2 * np.sum(np.power(weights,2))
        
        #Three hidden layors
        nn = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = f_train.shape[1], act_fn = identity))
        weights = nn.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches, epochs = epochs, l2 = l2)
        z, a = nn.feed_forward(X_test)
        
        MSE[2,i] = CostOLS.func(f_test,a[-1]) + l2 * np.sum(np.power(weights,2))
        
        i += 1
        
    fig, ax = plt.subplots(figsize=(10, 5))
    MSE[MSE > 10e1] = np.nan
    heatmap = sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='.4f')
    ax.set_xlabel("Neurons")
    ax.set_ylabel("Hidden Layers")
    ax.set_xticklabels(neurons)
    ax.set_yticklabels(layors)
    heatmap.set_facecolor('xkcd:grey')
    plt.tight_layout()
    if save == "Y": 
        plt.tight_layout()
        fig.savefig("../results/figures/Regression/NN_reg_sigmoid_neurons_layers.pdf")
    else:
        plt.show()

    print("[DONE]\n")

def plot_epochs_vs_batches(neurons=15, h_layers=1, eta=0.1, l2=0.0):
    print("MSE for different number of batches and operations:...")
    
    batches = np.power(2,np.arange(4,10)).astype("int")
    numbofit = (batches[-1]*np.arange(5,12))
    
    MSE = np.zeros((len(batches),len(numbofit)))
    
    nn = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
    nn.add_layer(layer(neurons = neurons, act_fn = sigmoid))
    nn.add_layer(layer(neurons = f_train.shape[1], act_fn = identity))
    
    i = 0
    for ba in batches:
        j = 0
        for it in numbofit:
            nn.reset()
            weights = nn.train(X_train = X_train, f_train = f_train, 
                               eta=eta, batches = ba ,epochs = int(it/ba), l2 = l2)
            z, a = nn.feed_forward(X_test)
            
            MSE[i,j] = CostOLS.func(f_test,a[-1]) + l2 * np.sum(np.power(weights, 2))
            
            j += 1
        i += 1
        
    fig, ax = plt.subplots(figsize=(10, 5))
    MSE[MSE > 10e1] = np.nan
    heatmap = sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='.4f')
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Batches")
    ax.set_xticklabels(numbofit)
    ax.set_yticklabels(batches)
    heatmap.set_facecolor('xkcd:grey')
    if save == "Y": 
        plt.tight_layout()
        fig.savefig("../results/figures/Regression/NN_reg_sigmoid_iterations_batches.pdf")
    else:
        plt.show()    
    print("[DONE]\n")

def plot_lambda_vs_eta(neurons=15, h_layers=1, batches=512, epochs=10, func = sigmoid):
    print("MSE for different number of l2 parameters and learningrates:...")
    
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
    test = sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='.4f')
    ax.set_xlabel("l2 parameter")
    ax.set_ylabel("learningrate")
    ax.set_xticklabels(l2s)
    ax.set_yticklabels(etas)
    # if type(func) is type:
    #     ax.set_title(f"NN with {func.__name__}")
    # else: 
    #     ax.set_title(f"NN with {func.__class__.__name__}")
    test.set_facecolor('xkcd:grey')
    if save == "Y": 
        plt.tight_layout()
        if type(func) is type:
            fig.savefig(f"../results/figures/Regression/NN_reg_{func.__name__}_l2_eta.pdf")
        else: 
            fig.savefig(f"../results/figures/Regression/NN_reg_{func.__class__.__name__}_l2_eta.pdf")
    else:
        plt.show()
    print("[DONE]\n")

def plot_bias(batches = 512, neurons = 15, eta = 0.1, l2=0.0, n_epochs=12):
    print("MSE for different bias inits:...")

    epochs = range(1, n_epochs, 20)
    
    MSE = np.zeros((3,len(epochs)))
    
    nn1 = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
    nn1.add_layer(layer(neurons = neurons, act_fn = tanh))
    nn1.add_layer(layer(neurons = f_train.shape[1], act_fn = identity))
    
    nn2 = FFNN(parameters=X_train.shape[1], cost_fn = CostOLS)
    nn2.add_layer(layer(neurons = neurons, act_fn = tanh, createbias="ones"))
    nn2.add_layer(layer(neurons = f_train.shape[1], act_fn = tanh, createbias="ones"))
    
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
    if save == "Y": 
        plt.savefig(f"../results/figures/Regression/NN_reg_bias.pdf")
    else:
        plt.show()    
    print("[DONE]\n")


# Initializations from SGD analysis 
plot_neurons_vs_layers(batches=64, epochs=300, eta=0.1, l2=0.0000)
plot_epochs_vs_batches(neurons=30, h_layers=3, eta=0.1, l2=0.0000)
plot_lambda_vs_eta(neurons=30, h_layers=3, batches=64, epochs=300, func = sigmoid)
plot_lambda_vs_eta(neurons=30, h_layers=3, batches=64, epochs=300, func = relu)
plot_lambda_vs_eta(neurons=30, h_layers=3, batches=64, epochs=300, func = leaky_relu(hyperpar = 0.01))
# plot_lambda_vs_eta(neurons=15, h_layers=1, batches=512, epochs=10, func = tanh)
plot_bias(batches = 64, neurons = 30, eta = 0.1, l2=0.0, n_epochs=300)


#Tracking Code: 8 6 7 9 2 9 0 1 9 6
