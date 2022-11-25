#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
@author: simon
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from SGD import StochGradDecent


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

#rescale
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

epochs = 200
batches = 16

etas = [1e0,1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
n = 7
l2s = np.zeros(n)
l2s[:n-1] = np.power(10.0,1-1*np.arange(n-1))

accuracy = np.zeros((len(etas),len(l2s)))

i = 0
for eta in etas:
    j = 0
    for l2 in l2s:
        sd = StochGradDecent(X_train, f_train, cost_fn=cross_entropy(l2 = l2))
        beta = sd.const(epochs = epochs, batches = batches, learningrate = eta)
        prediction = (1.0 / (1.0+np.exp(X_test@beta))).round()
        accuracy[i,j] = np.sum(prediction == f_test)/f_test.shape[0]
        j += 1
    i += 1

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(accuracy, annot=True, ax=ax, cmap="viridis_r", cbar_kws={'label': 'MSE'}, fmt='1.3e')
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel(r"$\eta$")
ax.set_xticklabels(l2s)
ax.set_yticklabels(etas)

#Get breast cancer data and split it into test/train sets
cancer = load_breast_cancer()
data = cancer.data
target = cancer.target.reshape(-1,1)

X_train, X_test, f_train, f_test = train_test_split(data, target, test_size=0.2, random_state=1)

logreg = LogisticRegression(solver='lbfgs', max_iter = epochs)
logreg.fit(X_train, f_train)
accuracy_sl = logreg.score(X_test,f_test)
time.sleep(10)

print("Test set accuracy with Logistic Regression:",accuracy_sl)
