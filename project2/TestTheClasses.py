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
from functions import costfunctions
import time


dimension = 1
coef = [3.5,3,4]
order = 2

func = functions(order = order, dimension=dimension, sigma=0.0,
                 coef=coef,points= 20000)
costfunc = costfunctions.CostOLS_beta

data, funcval = func.polynomial()
poly = PolynomialFeatures(degree=order)
X = poly.fit_transform(data)

reg = LinearRegression(X,funcval)
gd = GradDecent(X,funcval,costfunc=costfunc)
sd = StochGradDecent(X,funcval,costfunc=costfunc)


beta1 = reg.ols()
start = time.time()
beta2 = gd.const()
beta3 = gd.momentum()
beta4 = gd.adagrad()
beta5 = gd.adagrad(momentum = True,learningrate = 1)
beta6 = gd.rmsprop()
beta7 = gd.adam()
end = time.time()

print(f'Calc time GD: {end-start}')


start = time.time()
beta8 =  sd.const()
beta9 = sd.momentum()
beta10 = sd.adagrad()
beta11 = sd.adagrad(momentum = True,learningrate = 1)
beta12 = sd.rmsprop()
beta13 = sd.adam()
end = time.time()

print(f'Calc time SGD: {end-start}')

print("Gradient Decent")
print(f'OLS: \n {beta1}')
print(f' const: \n {beta2}')
print(f'momentum: \n {beta3}')
print(f'adagrad without momentum: \n {beta4}')
print(f'adagrad with momentum: \n {beta5}')
print(f'RMSprop: \n {beta6}')
print(f' adam: \n {beta7}')


print("Stochastic Gradient decent")
print(f'OLS: \n {beta1}')
print(f' const: \n {beta8}')
print(f'momentum: \n {beta9}')
print(f'adagrad without momentum: \n {beta10}')
print(f'adagrad with momentum: \n {beta11}')
print(f'RMSprop: \n {beta12}')
print(f' adam: \n {beta13}')


plt.scatter(data,funcval)