#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
@author: simon

This script generates every plot for analysis of the Wisconson breast cancer data set
with the help of a FFNN.
RuntimeWarning: overflow are normal since we get into regions were the NN does not converge!
We did choose the sigmoid function as the output activation function everytime.
Here is a quick overview what each function does:

plot_neurons_vs_layers(batches, epochs, eta, l2)
    Plots the accuracy against different number of layors of hidden layor 
    and neurons (per hidden layor)
    
plot_epochs_vs_batches(neurons, h_layers, eta, l2)
    Plots the accuracy against different number of epochs and number of operations

plot_lambda_vs_eta(neurons, h_layers, batches, epochs, hfunc, ofunc, title)
    Plots the accuracy for sigmoid act. function against different l2 and learningrate values
    
plot_bias(batches, neurons, eta, l2)
    Creates a Plot were for differnt bias initializations the accuracy is plottet 
    against the number of epochs

To run a function comment in the call at the bottom of the script
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from FFNN import FFNN
from layer import layer
from FrankeFunction import FrankeFunction
from activation_functions import sigmoid,identity,relu,leaky_relu,tanh

# Save figures (Y/N)
save = "Y"

class cross_entropy:
    """"
    Defines the Cost function and its derivative used for all calculations in this script
    """
    def func(y,ytilde):
        funcval = np.sum(y*np.log(ytilde) + (1-y)*np.log(1-ytilde))
        return -1/y.size * np.sum(funcval)
    def grad(y,ytilde):
        return -1/y.size*(y/(ytilde)-(1-y)/(1-ytilde))

# Get breast cancer data and split it into test/train sets
cancer = load_breast_cancer()
data = cancer.data
target = cancer.target.reshape(-1,1)
X_train, X_test, f_train, f_test = train_test_split(data, target, test_size=0.2, random_state=1)

# rescale
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


def plot_neurons_vs_layers(batches=64, epochs=16, eta=0.1, l2=0.0):
    print("Accuracy for different number of hidden layors and neurons:...")
    
    layors = np.arange(1,4)
    neurons = np.arange(0,35,5)
    neurons[0] = 1
    
    accuracy = np.zeros((len(layors),len(neurons)))

    
    i = 0
    for nr in neurons:
        #One hidden layor
        nn = FFNN(parameters=X_train.shape[1], cost_fn = cross_entropy)
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = f_train.shape[1], act_fn = sigmoid))
        weights = nn.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches,epochs = epochs, l2 = l2)
        z, a = nn.feed_forward(X_test)
        
        prediction = a[-1].round()
        accuracy[0,i] = np.sum(prediction == f_test)/f_test.shape[0]
        
        
        #Two hidden layors
        nn = FFNN(parameters=X_train.shape[1], cost_fn = cross_entropy)
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = f_train.shape[1], act_fn = sigmoid))
        weights = nn.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches,epochs = epochs, l2 = l2)
        z, a = nn.feed_forward(X_test)


        prediction = a[-1].round()
        accuracy[1,i] = np.sum(prediction == f_test)/f_test.shape[0]
        
        #Three hidden layors
        nn = FFNN(parameters=X_train.shape[1], cost_fn = cross_entropy)
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = nr, act_fn = sigmoid))
        nn.add_layer(layer(neurons = f_train.shape[1], act_fn = sigmoid))
        weights = nn.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches,epochs = epochs, l2 = l2)
        z, a = nn.feed_forward(X_test)
        
        prediction = a[-1].round()
        accuracy[2,i] = np.sum(prediction == f_test)/f_test.shape[0]
        
        i += 1
        
    fig, ax = plt.subplots(figsize=(10, 5))
    accuracy[accuracy < 1e-3] = np.nan
    heatmap = sns.heatmap(accuracy, annot=True, ax=ax, cmap="viridis_r", cbar_kws={'label': 'accuracy'}, fmt='.4f')
    ax.set_xlabel("Neurons")
    ax.set_ylabel("Hidden Layers")
    ax.set_xticklabels(neurons)
    ax.set_yticklabels(layors)
    heatmap.set_facecolor('xkcd:grey')
    if save == "Y": 
        fig.savefig("../results/figures/Classification/NN_class_sigmoid_neurons_layers.pdf")
    else:
        plt.show()

    print("[DONE]\n")

def plot_epochs_vs_batches(neurons=10, h_layers=1, eta=0.1, l2=0.0):
    print("Accuracy for different number of batches and operations:...")
    
    batches = np.power(2,np.arange(5,9)).astype("int")
    numbofit = (batches[-1]*np.arange(1,9))
    
    accuracy = np.zeros((len(batches),len(numbofit)))
    
    nn = FFNN(parameters=X_train.shape[1], cost_fn = cross_entropy)
    nn.add_layer(layer(neurons = neurons, act_fn = sigmoid))
    nn.add_layer(layer(neurons = f_train.shape[1], act_fn = sigmoid))
    
    i = 0
    for ba in batches:
        j = 0
        for it in numbofit:
            nn.reset()
            weights = nn.train(X_train = X_train, f_train = f_train, 
                               eta=eta, batches = ba ,epochs = int(it/ba) , l2 = l2)
            z, a = nn.feed_forward(X_test)
            
            prediction = a[-1].round()
            accuracy[i,j] = np.sum(prediction == f_test)/f_test.shape[0]
            
            j += 1
        i += 1
        
    fig, ax = plt.subplots(figsize=(10, 5))
    accuracy[accuracy < 1e-3] = np.nan
    heatmap = sns.heatmap(accuracy, annot=True, ax=ax, cmap="viridis_r", cbar_kws={'label': 'accuracy'}, fmt='.4f')
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Batches")
    ax.set_xticklabels(numbofit)
    ax.set_yticklabels(batches)
    heatmap.set_facecolor('xkcd:grey')
    if save == "Y": 
        fig.savefig("../results/figures/Classification/NN_class_sigmoid_iterations_batches.pdf")
    else:
        plt.show()    
    print("[DONE]\n")

def plot_lambda_vs_eta(neurons=10, h_layers=1, batches=256, epochs=7, \
    hfunc = sigmoid, ofunc = sigmoid, title = None):
    print("Accuracy for different number of l2 parameters and learningrates:...")
    
    etas = [1e0,1e-1, 1e-2, 1e-3, 1e-4]
    n = 7
    l2s = np.zeros(n)
    l2s[:n-1] = np.power(10.0,1-1*np.arange(n-1))
    
    accuracy = np.zeros((len(etas),len(l2s)))
    
    nn = FFNN(parameters=X_train.shape[1], cost_fn = cross_entropy)
    nn.add_layer(layer(neurons = neurons, act_fn = hfunc))
    nn.add_layer(layer(neurons = f_train.shape[1], act_fn = ofunc))
    
    i = 0
    for eta in etas:
        j = 0
        for l2 in l2s:
            nn.reset()
            weights = nn.train(X_train = X_train, f_train = f_train, 
                               eta=eta, batches = batches ,epochs = epochs, l2 = l2)
            z, a = nn.feed_forward(X_test)
            
            prediction = a[-1].round()
            accuracy[i,j] = np.sum(prediction == f_test)/f_test.shape[0]
            
            j += 1
        i += 1
    
    fig, ax = plt.subplots(figsize=(10, 5))
    accuracy[accuracy < 1e-3] = np.nan
    heatmap = sns.heatmap(accuracy, annot=True, ax=ax, cmap="viridis_r", cbar_kws={'label': 'accuracy'}, fmt='.4f')
    ax.set_xlabel(r"\lambda")
    ax.set_ylabel(r"\eta")
    ax.set_xticklabels(l2s)
    ax.set_yticklabels(etas)
    heatmap.set_facecolor('xkcd:grey')
    if save == "Y": 
        if title is not None:
            fig.savefig(f"../results/figures/Classification/NN_class_{title}_l2_eta.pdf")
    else: 
        plt.show()
    print("[DONE]\n")

def plot_bias(batches = 7, neurons = 10, eta = 0.1, l2=0.0):
    print("accuracy for different bias inits:...")
   
    epochs = range(1,12)
    
    accuracy = np.zeros((3,len(epochs)))
    
    nn1 = FFNN(parameters=X_train.shape[1], cost_fn = cross_entropy)
    nn1.add_layer(layer(neurons = neurons, act_fn = tanh))
    nn1.add_layer(layer(neurons = f_train.shape[1], act_fn = sigmoid))
    
    nn2 = FFNN(parameters=X_train.shape[1], cost_fn = cross_entropy)
    nn2.add_layer(layer(neurons = neurons, act_fn = tanh, createbias="ones"))
    nn2.add_layer(layer(neurons = f_train.shape[1], act_fn = sigmoid, createbias="ones"))
    
    nn3 = FFNN(parameters=X_train.shape[1], cost_fn = cross_entropy)
    nn3.add_layer(layer(neurons = neurons, act_fn = tanh, createbias="zeros"))
    nn3.add_layer(layer(neurons = f_train.shape[1], act_fn = sigmoid, createbias="zeros"))
    
    i = 0
    for epoch in epochs:
        #random bias
        nn1.reset()
        weights1 = nn1.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches ,epochs = epoch, l2 = l2)
        z1, a1 = nn1.feed_forward(X_test)
        
        prediction1 = a1[-1].round()
        accuracy[0,i] = np.sum(prediction1 == f_test)/f_test.shape[0]
        
        #bias is 1
        nn2.reset()
        weights2 = nn2.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches ,epochs = epoch, l2 = l2)
        z2, a2 = nn2.feed_forward(X_test)
        
        prediction2 = a2[-1].round()
        accuracy[1,i] = np.sum(prediction2 == f_test)/f_test.shape[0]
        
        #bias is 0
        nn3.reset()
        weights3 = nn3.train(X_train = X_train, f_train = f_train, 
                           eta=eta, batches = batches ,epochs = epoch, l2 = l2)
        z3, a3 = nn3.feed_forward(X_test)
        
        prediction3 = a3[-1].round()
        accuracy[2,i] = np.sum(prediction3 == f_test)/f_test.shape[0]
        
        i += 1
    
    plt.plot(epochs,accuracy[0,:], label = "rand")
    plt.plot(epochs,accuracy[1,:], label = "ones")
    plt.plot(epochs,accuracy[2,:], label = "zeros")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    if save == "Y": 
        plt.savefig(f"../results/figures/Classification/NN_class_bias.pdf")
    else:
        plt.show()    
    print("[DONE]\n")


plot_neurons_vs_layers(batches=64, epochs=16, eta=0.1, l2=0.0)
plot_epochs_vs_batches(neurons=10, h_layers=1, eta=0.1, l2=0.0)
plot_lambda_vs_eta(neurons=10, h_layers=1, batches=256, epochs=7, hfunc = sigmoid, ofunc = sigmoid, title = "sigmoid")
"""Plots the accuracy for tanh as hidden act.function and sigmoid as tanh act.func
    against different l2 and learningrate values"""
plot_lambda_vs_eta(neurons=10, h_layers=1, batches=256, epochs=7, hfunc = tanh, ofunc = sigmoid, title = "tanh_sig")
plot_bias(batches = 7, neurons = 10, eta = 0.1, l2=0.0)

"""
The following do not converge
"""
# plot_lambda_vs_eta(neurons=10, h_layers=1, batches=256, epochs=7, hfunc = tanh, ofunc = tanh, title = "tanh_sig")
# plot_lambda_vs_eta(neurons=10, h_layers=1, batches=256, epochs=7, hfunc = relu, ofunc = sigmoid, title = "tanh_sig")
# plot_lambda_vs_eta(neurons=10, h_layers=1, batches=256, epochs=7, hfunc = leaky_relu(hyperpar = 0.001), ofunc = sigmoid, title = "tanh_sig")
