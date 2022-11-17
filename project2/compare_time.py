#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

from sklearn.preprocessing import PolynomialFeatures
from Grad_Decent import GradDecent
from S_Grad_Decent import StochGradDecent
from gen_data import functions
from functions import costfunctions
import time
import numpy as np

func = functions(dimension=2, sigma=0.25 ,points= 100)
costfunc = costfunctions.CostOLS_beta


data, funcval = func.FrankeFunction()

order = 4
poly = PolynomialFeatures(degree=order)
X = poly.fit_transform(data)

epochs = 200
batches = 100
lr = 0.1
gd = GradDecent(X,funcval,costfunc=costfunc)
sd = StochGradDecent(X,funcval,costfunc=costfunc)
times = np.zeros(2)

start = time.time()
beta_gd = gd.const(iterations = epochs, learningrate = lr)
end = time.time()
times[0] = end-start

start = time.time()
beta_sd = sd.const(epochs = epochs, batches = batches, learningrate = lr)
end = time.time()
times[1] = end-start

size = np.array([1,batches])
with np.printoptions(formatter={'float': lambda x: format(x, '6.3e')}):
    print("The first list element corresponds to GD and the second one to SGD")
    print(f"time per epoch in s: \t \t {times/epochs}")
    print(f"time per operation in s: \t {times/(epochs*size)}")
