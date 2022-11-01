#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:33:41 2022

@author: simon
"""

import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from Linear_Regression import LinearRegression
from Grad_Decent import GradDecent
from S_Grad_Decent import StochGradDecent
from gen_data import functions

def CostOLS(y,X,beta):
    return 1/X.shape[0]*((y-X@beta).T@(y-X@beta))

dimension = 1
coef = [3.5,3,4]
order = 2

func = functions(order = order, dimension=dimension, sigma=0.0,
                 coef=coef,points= 2000)

data, funcval = func.polynomial()
poly = PolynomialFeatures(degree=order)
X = poly.fit_transform(data)

reg = LinearRegression(X,funcval)
gd = GradDecent(X,funcval,CostOLS)
sd = StochGradDecent(X,funcval,CostOLS)

beta1 = reg.ols()
beta2 = gd.const()
beta3 = gd.momentum()
beta4 = gd.adagrad()
beta5 = gd.adagrad(momentum = True,learningrate = 1)
beta6 = gd.rmsprop()
beta7 = gd.adam()

print(beta1)
print(beta2)
print(beta3)
print(beta4)
print(beta5)
print(beta6)
print(beta7)