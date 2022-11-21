#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import numpy as np

class costfunctions:
    class CostOLS_beta:
        def __init__(self,y,X,beta):
            self.y = y
            self.X = X
            self.beta = beta
        
            self.XBeta = X@beta
            self.XT = X.T
            self.n = X.shape[0]
        
        
        def func(self):
            return 1/self.n*((self.y-self.XBeta).T@(self.y-self.XBeta))
            
        def derivative(self):
            return -2/self.n*(self.XT@self.y-self.XT@self.XBeta)
    
    class CostOLS:
        def __init__(self,y,ytilde):
            self.y = y
            self.ytilde = ytilde
            
        def func(self):
            return np.mean(np.matmul((self.y-self.ytilde).T,(self.y-self.yilde)))
        def derivative(self):
            return (2/self.ytilde.shape[0])*(self.ytilde-self.y)
    
    class Ridge_beta:
        def __init__(self,y,X,beta,hyperpar):
            self.y = y
            self.X = X
            self.beta = beta
            self.hyperpar = hyperpar
        
            self.XBeta = X@beta
            self.XT = X.T
            self.n = X.shape[0]
        
        def func(self):
            return
        
        def derivative(self):
            return -2/self.n*self.XT @ (self.X @ self.beta - self.y) + 2*self.hyperpar
            
    

class activation_functions:
    """Class for different activation functions """
    
    def sigmoid(self):
        x = self.x
        return 1/(1 + np.exp(-x)), 1/(1 + np.exp(-x))*(1-1/(1 + np.exp(-x)))
    
    class identity():
        def __init__(self,x) -> None:
            self.x = x
        def func(self): return self.x
        def derivative(self): return np.ones(self.x.shape)
    
    def probability(self):
        x = self.x
        return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
