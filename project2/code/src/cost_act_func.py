#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import numpy as np
        


class CostOLS_beta:
    def func(y,X,beta):
        XBeta = X@beta
        return 1/X.shape[0]*((y-XBeta).T@(y-XBeta))
        
    def derivative(y,X,beta):
        XT = X.T
        return -2/X.shape[0]*(XT @ (y-X@beta))

#we split ridge for faster calc. for hyperpar = 0
class Ridge_beta:
    def __init__(self,hyperpar = 0.0):
        self.hyperpar = hyperpar
    def func(self,y,X,beta):
        XBeta = X@beta
        return 1/X.shape[0]*((y-XBeta).T@(y-XBeta)) + self.hyperpar * (beta.T@beta)
    def derivative(self,y,X,beta):
        XT = X.T
        return -2/X.shape[0]*(XT @ (y-X@beta)) + 2*self.hyperpar*beta

class CostOLS:     
    def func(y,ytilde):
        return np.mean(np.power((y-ytilde),2))
    def derivative(y,ytilde):
        return (2/ytilde.shape[0])*(ytilde-y)
    
class cross_entropy:
    def func(y,ytilde):
        funcval = np.sum(y*np.log(ytilde) + (1-y)*np.log(1-ytilde))
        return -1/y.size * np.sum(funcval)
    def derivative(y,ytilde):
        return -1/y.size*(y/(ytilde)-(1-y)/(1-ytilde))
            
    

class activation_functions:
    """Class for different activation functions """
    
    class sigmoid:
        def func(x): 
            return 1/(1 + np.exp(-x))
        def derivative(x):
            exp_x = np.exp(-x)
            return 1/(1 + exp_x)*(1-1/(1 + exp_x))
    
    class identity:
        def func(x): return x
        def derivative(x): return np.ones(x.shape)
        
    class relu:
        def func(x): 
            return np.maximum(0,x)
        def derivative(x):
            return np.heaviside(x,0)
    
    class leaky_relu:
        def __init__(self,hyperpar = 0.01):
            self.hyperpar = hyperpar

        def func(self,x): 
            var = x.copy()
            var[var<0] = self.hyperpar*var[var<0]
            return var
        def derivative(self,x):
            var = x.copy()
            var[var<0] = self.hyperpar
            var[var>0] = 1
            return var