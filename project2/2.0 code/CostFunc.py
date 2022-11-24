#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

"""
NOTE: i want to plug cost functions into the scrips generating the plots later and remove this.
"""

import numpy as np
        


class CostOLS_beta:
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
    def grad(y,ytilde):
        return (2/ytilde.shape[0])*(ytilde-y)
    
class cross_entropy:
    def func(y,ytilde):
        funcval = np.sum(y*np.log(ytilde) + (1-y)*np.log(1-ytilde))
        return -1/y.size * np.sum(funcval)
    def grad(y,ytilde):
        return -1/y.size*(y/(ytilde)-(1-y)/(1-ytilde))
            