#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import numpy as np
    
class sigmoid:
    def func(x): 
        return 1/(1 + np.exp(-x))
    def grad(x):
        exp_x = np.exp(-x)
        return 1/(1 + exp_x)*(1-1/(1 + exp_x))

class identity:
    def func(x): return x
    def grad(x): return np.ones(x.shape)
    
class relu:
    def func(x): 
        return np.maximum(0,x)
    def grad(x):
        return np.heaviside(x,0)

class leaky_relu:
    def __init__(self,hyperpar = 0.01):
        self.hyperpar = hyperpar

    def func(self,x): 
        var = x.copy()
        var[var<0] = self.hyperpar*var[var<0]
        return var
    def grad(self,x):
        var = x.copy()
        var[var<0] = self.hyperpar
        var[var>0] = 1
        return var