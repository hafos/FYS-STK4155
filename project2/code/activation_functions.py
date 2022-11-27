#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import numpy as np
    
class sigmoid:
    """
    function and derivative of the sigmoid function
    """
    def func(x): 
        return 1.0/(1.0 + np.exp(-x))
    def grad(x):
        exp_x = np.exp(-x)
        return 1.0/(1.0 + exp_x)*(1.0-1.0/(1.0 + exp_x))

class identity:
    """
    function and derivative of the identity function
    """
    def func(x): return x
    def grad(x): return np.ones(x.shape)
    
class relu:
    """
    function and derivative of the relu function
    """
    def func(x): 
        return np.maximum(0.0,x)
    def grad(x):
        return np.heaviside(x,0.0)

class leaky_relu:
    """
    function and derivative of the leaky relu function
    """
    def __init__(self,hyperpar = 0.01):
        """
        Arguments
        
        ---------
        hyperpar: float
            Define the constant of leaky relu (default: 0.01)
        """
        
        self.hyperpar = hyperpar

    def func(self,x): 
        var = x.copy()
        var[var<0] = self.hyperpar*var[var<0]
        return var
    def grad(self,x):
        var = x.copy()
        var[var<0] = self.hyperpar
        var[var>0] = 1.0
        return var

class tanh:
    """
    function and derivative of tanh
    """
    def func(x): return np.tanh(x)
    def grad(x): return 1.0 / (1.0+np.power(x,2))