#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon

This script generates everything for fitting the Franke function with GD and SGD.
Here is a quick overview what each function does:

part_a():
    MSE against number of operations for GD with different learningrates
    
part_b()
    MSE for GD for different l2 parameters and different learningrates

part_c()
    MSE against number of operations for GD with different learning rate update algorithms
    
part_d()
    Compare comp. time of GD and SGD
    
part_e():
    SGD for different batches and number of operations (~epochs)
    
part_f()
    MSE against number of operations for SGD with different learning rate update algorithms

To run a function comment in the call at the bottom of the script
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from SGD import StochGradDecent
from FrankeFunction import FrankeFunction

# Save figures (Y/N)
save = "Y"

class CostOLS:     
    """"
    Defines the Cost function and its derivative used for all calculations in this script
    """
    def __init__(self,hyperpar = 0.0):
        self.hyperpar = hyperpar
    def func(self,y,X,beta):
        XBeta = X@beta
        return 1/X.shape[0]*((y-XBeta).T@(y-XBeta)) + self.hyperpar * (beta.T@beta)
        
    def grad(self,y,X,beta):
        XT = X.T
        return -2/X.shape[0]*(XT @ (y-X@beta)) + 2*self.hyperpar*beta
    
#Generate data, the design matrix and split it into test/train sets
data, funcval = FrankeFunction(points = 100, sigma=0.25)

order = 7
poly = PolynomialFeatures(degree=order)
data = poly.fit_transform(data)

X_train, X_test, f_train, f_test = train_test_split(data, funcval, test_size=0.2, random_state=1)

def plot_epochs(n_epoch=300):
    print("GD for different learningrates (eta):...", end = "")
    
    cost_fn = CostOLS()
    sgd = StochGradDecent(X_train, f_train, cost_fn = cost_fn)
    
    etas = [1e-1, 1e-2, 1e-3]
    epochs = np.arange(n_epoch)
 
    MSE = np.zeros((len(etas),len(epochs)))

    i = 0 
    for eta in etas:
        j = 0
        for epoch in epochs:
            beta = sgd.const(epochs = epoch, batches = 1, learningrate = eta)
            MSE[i,j] = cost_fn.func(f_test, X_test, beta)
            j +=1
        i+=1

    for i in range(len(etas)):
        plt.plot(epochs, MSE[i,:], label = fr"GD with $\eta$ = {etas[i]}")
        plt.legend()
    # plt.tight_layout()
    plt.ylabel("MSE")
    plt.xlabel("epochs")
    if save == "Y": 
        plt.savefig(f"../results/figures/Regression/SGD_epochs.pdf")
    else:
        plt.show()    
    print("[DONE]\n")
    
def plot_lambda_vs_learningrates(epochs=500):
    print("MSE for different learningrates and l2 parameters for GD:...")
        
    etas = [1e-1, 1e-2, 1e-3, 1e-4]
    n = 5
    l2s = np.zeros(n)
    l2s[:n-1] = np.power(10.0,-1*np.arange(1,n))

    MSE = np.zeros((len(etas),len(l2s)))

    i = 0
    for eta in etas:
        j = 0
        for l2 in l2s:
            cost_fn = CostOLS(hyperpar=l2)
            sgd = StochGradDecent(X_train, f_train, cost_fn = cost_fn)
            beta = sgd.const(epochs = epochs, batches = 1, learningrate = eta)
            MSE[i,j] = cost_fn.func(f_test, X_test, beta)
            j += 1
        i += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='.4f')
    ax.set_xlabel(r"l2 parameter")
    ax.set_ylabel(r"$\eta$")
    ax.set_xticklabels(l2s)
    ax.set_yticklabels(etas)
    if save == "Y": 
        plt.savefig(f"../results/figures/Regression/SGD_MSE_lambda_eta.pdf")
    else:
        plt.show()    
    print("[DONE]\n")

def plot_MSE_learningrates_GD_extra(n_epochs=200, eta=1e-1):
    print("compare different learningrate upgrade methodes for GD:...")
    
    cost_fn = CostOLS()
    sgd = StochGradDecent(X_train, f_train, cost_fn=cost_fn)

    epochs = np.arange(0, n_epochs)
    #compare for one higher number of operations
    epochs = np.append(epochs, 1024)
    
    #adaptive is not included since it performed worse then every other methode in previous tests
    methodes = ["GD","GD with momentum","GD Adagrad", "GD Adagrad with momentum","GD RMS","GD ADAM"]

    # Initialize and calculate the MSE 
    MSE = np.zeros((len(methodes),len(epochs)))
    beta = np.zeros((len(methodes),X_train.shape[1]))
    
    i = 0
    for epoch in epochs:
        # Initialize betas 
        beta[0,:] = sgd.const(epochs = epoch, batches = 1, learningrate = eta).ravel()
        beta[1,:] = sgd.momentum(epochs = epoch, batches = 1, learningrate = eta).ravel() 
        beta[2,:] = sgd.adagrad(epochs = epoch, batches = 1, learningrate = eta).ravel() 
        beta[3,:] = sgd.adagrad(momentum = True, epochs = epoch, batches = 1, learningrate = eta).ravel() 
        beta[4,:] = sgd.rmsprop(epochs = epoch, batches = 1, learningrate = eta).ravel() 
        beta[5,:] = sgd.adam(epochs = epoch, batches = 1, learningrate = eta).ravel() 
        
        for j in range(len(methodes)):
            MSE[j,i] = cost_fn.func(f_test,X_test,beta[j,:].reshape(len(beta[j,:]),1))
            j += 1
        i += 1
    
    j = 0
    for name in methodes:
        plt.plot(epochs[5:-1], MSE[j,5:-1], label = name)
        j += 1
    
    plt.ylabel("MSE")
    plt.xlabel("number of operations")
    plt.legend(prop={'size': 8})
    if save == "Y": 
        plt.savefig(f"../results/figures/Regression/GD_MSE_learningrates_extra.pdf")
    else:
        plt.show()
    
    j = 0
    print(f"MSE for GD after {epochs[-2]} iterations for")
    with np.printoptions(formatter={'float': lambda x: format(x, '6.3e')}):
        for name in methodes:
            print(f"{name}: {MSE[j,-2]}")
            j += 1
    
    print("\n")
    
    j = 0
    print(f"MSE for GD after {epochs[-1]+1} iterations for")
    with np.printoptions(formatter={'float': lambda x: format(x, '6.3e')}):
        for name in methodes:
            print(f"{name}: {MSE[j,-1]}")
            j += 1

    print("[DONE]\n")

def time_GD_vs_SGD(batches=4, epochs=200, eta=0.1):
    print("compare calc. time for GD and SGD:...")

    cost_fn = CostOLS()
    sgd = StochGradDecent(X_train, f_train, cost_fn=cost_fn)
    
    times = np.zeros(2)

    start = time.time()
    beta_gd = sgd.const(epochs = epochs, batches = 1, learningrate = eta)
    end = time.time()
    times[0] = end-start

    start = time.time()
    beta_sd = sgd.const(epochs = epochs, batches = batches, learningrate = eta)
    end = time.time()
    times[1] = end-start

    size = np.array([1, batches])
    with np.printoptions(formatter={'float': lambda x: format(x, '6.3e')}):
        print(f"The first list element corresponds to GD and the second one to SGD with {batches} batches")
        print(f"time per epoch in s: \t \t {times/epochs}")
        print(f"time per operation in s: \t {times/(epochs*size)}")
        print("[DONE]\n")

def plot_operations_vs_batches(eta=1e-1): 
    print("SGD for different number of operations and batches:...", end = "")
    
    cost_fn = CostOLS()
    sgd = StochGradDecent(X_train, f_train, cost_fn = cost_fn)

    batches = np.power(2,np.arange(0,8)).astype("int")
    numbofit = (batches[-1]*np.arange(1,9))

    # Initialize and calculate the MSE 
    MSE = np.zeros((len(batches),len(numbofit)))
    
    i = 0
    for batch in batches:
        j = 0
        for it in numbofit:
            beta = sgd.const(epochs = int(it/batch), batches = batch, learningrate = eta)
            MSE[i,j] = cost_fn.func(f_test, X_test, beta)
            j +=1
        i += 1
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis_r", cbar_kws={'label': 'MSE'}, fmt='.4f')
    ax.set_xlabel("Number of Operations")
    ax.set_ylabel("Number of Batches")
    ax.set_xticklabels(numbofit)
    ax.set_yticklabels(batches)
    if save == "Y": 
        plt.savefig(f"../results/figures/Regression/SGD_op_batch.pdf")
    else:
        plt.show()

def plot_MSE_learningrates_SGD_extra(batches=64, eta=1e-1):
    print("compare different learningrate upgrade methodes for SGD:...")
    
    cost_fn = CostOLS()
    sgd = StochGradDecent(X_train, f_train, cost_fn=cost_fn)
    
    epochs = np.arange(0,16)

    methodes = ["SGD","SGD adaptive","SGD with momentum","SGD Adagrad", "SGD Adagrad with momentum","SGD RMS","SGD ADAM"]

    # Initialize and calculate the MSE 
    MSE = np.zeros((len(methodes),len(epochs)))
    beta = np.zeros((len(methodes),X_train.shape[1]))
    
    i = 0
    for epoch in epochs:
        # Initialize betas 
        beta[0,:] = sgd.const(epochs = epoch, batches = batches, learningrate = eta).ravel()
        beta[1,:] = sgd.adaptive(epochs = epoch, batches = batches, t_0 = eta).ravel()
        beta[2,:] = sgd.momentum(epochs = epoch, batches = batches, learningrate = eta).ravel() 
        beta[3,:] = sgd.adagrad(epochs = epoch, batches = batches, learningrate = eta).ravel() 
        beta[4,:] = sgd.adagrad(momentum = True, epochs = epoch, batches = batches, learningrate = eta).ravel() 
        beta[5,:] = sgd.rmsprop(epochs = epoch, batches = batches, learningrate = eta).ravel() 
        beta[6,:] = sgd.adam(epochs = epoch, batches = batches, learningrate = eta).ravel() 
        
        for j in range(len(methodes)):
            MSE[j,i] = cost_fn.func(f_test,X_test,beta[j,:].reshape(len(beta[j,:]),1))
            j += 1
        i += 1
    
    j = 0
    for name in methodes:
        plt.plot(epochs[1:]*batches, MSE[j,1:], label = name)
        j += 1
    
    plt.ylabel("MSE")
    plt.xlabel("Number of Operations")
    plt.legend(prop={'size': 8})
    if save == "Y": 
        plt.savefig(f"../results/figures/Regression/SGD_MSE_learningrates_extra.pdf")
    else:
        plt.show()
    
    j = 0
    print(f"MSE for SGD after {(epochs[-9,]+1)*batches} operations for")
    with np.printoptions(formatter={'float': lambda x: format(x, '6.3e')}):
        for name in methodes:
            print(f"{name}: {MSE[j,epochs[-9]]}")
            j += 1

    print("\n")
    
    j = 0
    print(f"MSE for GD after {(epochs[-1]+1)*batches} operations for")
    with np.printoptions(formatter={'float': lambda x: format(x, '6.3e')}):
        for name in methodes:
            print(f"{name}: {MSE[j,-1]}")
            j += 1

    print("[DONE]\n")

plot_epochs(n_epoch=300)
plot_lambda_vs_learningrates(epochs=500)
plot_MSE_learningrates_GD_extra(n_epochs=200, eta=1e-1)
time_GD_vs_SGD(batches=4, epochs=200, eta=0.1)
plot_operations_vs_batches(eta=1e-1)
plot_MSE_learningrates_SGD_extra(batches=64, eta=1e-1)