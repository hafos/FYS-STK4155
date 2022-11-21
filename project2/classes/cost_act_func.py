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
        return -2/X.shape[0]*(XT @ y-XT @ (X @ beta))
    
class CostOLS:     
    def func(y,ytilde):
        return np.mean(np.matmul((y-ytilde).T,(y-ytilde)))
    def derivative(y,ytilde):
        return (2/ytilde.shape[0])*(ytilde-y)
    
class Ridge_beta:
    def __init__(self,hyperpar = 0.0):
        self.hyperpar = hyperpar
    def func(self,y,X,beta):
        XBeta = X@beta
        return 1/X.shape[0]*((y-XBeta).T@(y-XBeta)) + self.hyperpar(beta.T@beta)
    def derivative(self,y,X,beta):
        return -2/X.shape[0]*X.T @ (X @ beta - y) + 2*self.hyperpar
            
    

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
