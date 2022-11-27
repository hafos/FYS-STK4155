#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon

This script generates everything for fitting the wisconsin breast cancer data with  SGD.
Here is a quick overview what each function does:

part_a():
    Accuracy for GD for different l2 parameters and different learningrates

part_b():
    Accuracy for GD with adam for different l2 parameters and different learningrates
    
    
part_c(eta):
    Accuracy forS GD with adam for different number of batches and number of 
    operations (~batches)
    
part_d()
    Accuracy with the help of scikit learns logistic regression

To run a function comment in the call at the bottom of the script
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from SGD import StochGradDecent

# Save figures (Y/N)
save = "Y"

class cross_entropy:     
    """"
    Defines the Cost function and its derivative used for all calculations in this script
    """
    def __init__(self,l2 = 0):
        self.l2 = l2
    def func(self,y,X,beta):
        return - (y * X@beta - np.log(1+np.exp(X@beta))) + self.l2 * np.sum(np.power(beta,2))
    def grad(self,y,X,beta):
        p = 1.0 / (1.0+np.exp(X@beta))
        return -X.T@(y-p) + 2*self.l2*beta


#Get breast cancer data and split it into test/train sets
cancer = load_breast_cancer()
data = cancer.data
target = cancer.target.reshape(-1,1)
X_train, X_test, f_train, f_test = train_test_split(data, target, test_size=0.2, random_state=1)

def plot_gd_lambda_vs_eta(epochs=1024, batches=1):
    print("Accuracy for different learningrates and l2 parameters for GD:...")
    X_train, X_test, f_train, f_test = train_test_split(data, target, test_size=0.2, random_state=1)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    etas = [1e0,1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    n = 7
    l2s = np.zeros(n)
    l2s[:n-1] = np.power(10.0,1-1*np.arange(n-1))
    
    accuracy = np.zeros((len(etas),len(l2s)))
    
    i = 0
    for eta in etas:
        j = 0
        for l2 in l2s:
            sgd = StochGradDecent(X_train, f_train, cost_fn=cross_entropy(l2 = l2))
            beta = sgd.const(epochs = epochs, batches = batches, learningrate = eta)
            prediction = (1.0 / (1.0+np.exp(-X_test@beta))).round()
            accuracy[i,j] = np.sum(prediction == f_test)/f_test.shape[0]
            j += 1
        i += 1
    
    fig, ax = plt.subplots(figsize=(10, 5))
    accuracy[accuracy < 5e-1] = np.nan
    heatmap = sns.heatmap(accuracy, annot=True, ax=ax, cmap="viridis_r", cbar_kws={'label': 'MSE'}, fmt='.4f')
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\eta$")
    ax.set_xticklabels(l2s)
    ax.set_yticklabels(etas)
    plt.title("Logistic regression with GD")
    heatmap.set_facecolor('xkcd:grey')
    if save == "Y": 
        plt.savefig(f"../results/figures/Classification/GD_acc_lambda_eta.pdf")
    else:
        plt.show()    
    print("[DONE]\n")

    
def plot_adam_lambda_vs_eta(epochs=1024, batches=1):
    print("Accuracy for different learningrates and l2 parameters for GD with adam:...")
    X_train, X_test, f_train, f_test = train_test_split(data, target, test_size=0.2, random_state=1)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    etas = [1e0,1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    n = 7
    l2s = np.zeros(n)
    l2s[:n-1] = np.power(10.0,1-1*np.arange(n-1))
    
    accuracy = np.zeros((len(etas),len(l2s)))
    
    i = 0
    for eta in etas:
        j = 0
        for l2 in l2s:
            sgd = StochGradDecent(X_train, f_train, cost_fn=cross_entropy(l2 = l2))
            beta = sgd.adam(epochs = epochs, batches = batches, learningrate = eta)
            prediction = (1.0 / (1.0+np.exp(-X_test@beta))).round()
            accuracy[i,j] = np.sum(prediction == f_test)/f_test.shape[0]
            j += 1
        i += 1
    
    fig, ax = plt.subplots(figsize=(10, 5))
    accuracy[accuracy < 5e-1] = np.nan
    heatmap = sns.heatmap(accuracy, annot=True, ax=ax, cmap="viridis_r", cbar_kws={'label': 'MSE'}, fmt='.4f')
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\eta$")
    ax.set_xticklabels(l2s)
    ax.set_yticklabels(etas)
    plt.title("Logistic regression with GD with adam")
    heatmap.set_facecolor('xkcd:grey')
    if save == "Y": 
        plt.savefig(f"../results/figures/Classification/adam_acc_lambda_eta.pdf")
    else:
        plt.show()    
    print("[DONE]\n")


def plot_operations_vs_batches(eta=0.0001, l2=0.0): 
    print("SGD for different number of operations and batches")
    print(f"with eta={eta}...")
    X_train, X_test, f_train, f_test = train_test_split(data, target, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
        
    batches = np.power(2,np.arange(0,8)).astype("int")
    numbofit = (batches[-1]*np.arange(1,9))
    
    sgd = StochGradDecent(X_train, f_train, cost_fn=cross_entropy(l2 = l2))

    # Initialize and calculate the MSE 
    accuracy = np.zeros((len(batches),len(numbofit)))
    
    i = 0
    for batch in batches:
        j = 0
        for it in numbofit:
            beta = sgd.adam(epochs = int(it/batch), batches = batch, learningrate = eta)
            prediction = (1.0 / (1.0+np.exp(-X_test@beta))).round()
            accuracy[i,j] = np.sum(prediction == f_test)/f_test.shape[0]
            j +=1
        i += 1
    
    fig, ax = plt.subplots(figsize=(10, 5))
    accuracy[accuracy < 1e-1] = np.nan
    heatmap = sns.heatmap(accuracy, annot=True, ax=ax, cmap="viridis_r", cbar_kws={'label': 'MSE'}, fmt='.4f')
    ax.set_xlabel("Number of Operations")
    ax.set_ylabel("Number of Batches")
    ax.set_xticklabels(numbofit)
    ax.set_yticklabels(batches)
    heatmap.set_facecolor('xkcd:grey')
    if save == "Y": 
        plt.savefig(f"../results/figures/Classification/SGD_operations_batches_{eta}.pdf")
    else:
        plt.show()    
    print("[DONE]\n")
    

def part_d(max_oper = 1024):
    print ("Get accuracy with the help of scikit learns logistic regression:...")
    X_train, X_test, f_train, f_test = train_test_split(data, target, test_size=0.2, random_state=1)

    logreg = LogisticRegression(solver='lbfgs', max_iter = max_oper)
    logreg.fit(X_train, f_train)
    accuracy_sl = logreg.score(X_test,f_test)
    time.sleep(2)
    
    print("\n \n \n")
    print("Test set accuracy with Logistic Regression:",accuracy_sl)
    print("[DONE]\n")

plot_gd_lambda_vs_eta(epochs=1024, batches=1)
plot_adam_lambda_vs_eta(epochs=1024, batches=1)
plot_operations_vs_batches(eta=0.0001, l2=0.0)

plot_operations_vs_batches(eta=0.001, l2=0.0)
plot_operations_vs_batches(eta=0.01, l2=0.0)
plot_operations_vs_batches(eta=0.1, l2=0.0)

part_d(max_oper = 1024)
