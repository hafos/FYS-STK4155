#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import numpy as np

class StochGradDecent:
    """Class for performing Stochastic Gradient Decent methods on a given dataset"""
    def __init__(self, X_train = None, f_train = None, cost_fn = None) -> None:
        """ 
        Constructor for generating an instance of the class.
        
        Arguments
        
        ---------
        X_train: array
            train data values (default: None)
        f_train: array
            train function values (default: None)
        cost_fn: class
            If one wants to use gradient decent, a cost function and the derivative
            has to be provided (default: None)
        """
        
        self.X_train = X_train
        self.f_train = f_train
        self.cost_fn= cost_fn
        np.random.seed(1999)
        self.beta = np.random.randn(X_train.shape[1],1)
        
    def const(self, epochs = int(10e2), batches = 10, learningrate= 10e-3):
        """
        Stochastic Gradient Decent with a constant learningrate
        
        Arguments
        ---------
        epochs: int
            Number of epochs (default: 10e2)  
        batches: int
            Number of batches (default: 10)
        learningrate: float
            learningrate (default: 10e-3)
        """
        
        X_train = self.X_train.copy()
        f_train = self.f_train.copy()
        beta = self.beta.copy()
        
        X_train = np.array_split(X_train,batches,axis=0)
        f_train = np.array_split(f_train,batches)
        
        np.random.seed(1999) #ensures reproducibility
        for itera in range(epochs):
            for i in range(batches):
                rd_ind = np.random.randint(batches)
                cost_fn = self.cost_fn
                gradient = cost_fn.grad(f_train[rd_ind],X_train[rd_ind],beta)
                beta -= learningrate*gradient
        
        return(beta)
    
    def adaptive(self, epochs = int(10e2), batches = 10, t_0 = 10e-1,
                 t_1=1.0):
        """
        Stochastic Gradient Decent with a adaptive learningrate
        
        Arguments
        ---------
        epochs: int
            Number of epochs (default: 10e2)  
        batches: int
            Number of batches (default: 10)
        t_0: float
            parameter 1 (default: 10e-3)
        t_1: float
            parameter 2 (default: 1.0)
        """
        
        X_train = self.X_train.copy()
        f_train = self.f_train.copy()
        beta = self.beta.copy()
        
        X_train = np.array_split(X_train,batches,axis=0)
        f_train = np.array_split(f_train,batches)
        
        np.random.seed(1999) #ensures reproducibility
        for itera in range(epochs):
            for i in range(batches):
                rd_ind = np.random.randint(batches)
                cost_fn = self.cost_fn
                gradient = cost_fn.grad(f_train[rd_ind],X_train[rd_ind],beta)
                learningrate = t_0/(t_1+itera)
                beta -= learningrate*gradient
        
        return(beta)
        
    
    def momentum(self, epochs = int(10e2), batches = 10, learningrate = 10e-1, 
                 delta_momentum = 0.3):
        """
        Momentum based Stochastic Gradient Decent
        
        Arguments
        ---------
        epochs: int
            Number of epochs (default: 10e2)  
        batches: int
            Number of batches (default: 10)
        learningrate: float
            Starting learningrate (default: 10e-2)
        delta_momentum: float
            momentum parameter (default: 0.3)           
        """
        
        X_train = self.X_train.copy()
        f_train = self.f_train.copy()
        beta = self.beta.copy()
        
        change = 0
        
        X_train = np.array_split(X_train,batches,axis=0)
        f_train = np.array_split(f_train,batches)
        
        np.random.seed(1999) #ensures reproducibility
        for itera in range(epochs):
            for i in range(batches):
                rd_ind = np.random.randint(batches)
                cost_fn = self.cost_fn
                gradient = cost_fn.grad(f_train[rd_ind],X_train[rd_ind],beta)
                new_change = learningrate*gradient+delta_momentum*change
                beta -= new_change
                change = new_change
        
        return beta 
    
    def adagrad(self, epochs = int(10e2), batches = 10, learningrate= 10e-1, 
                momentum = False, delta_momentum = 0.3):
        """
        Stochastic Gradient Decent with ADAGRAD
        
        Arguments
        ---------
        epochs: int
            Number of epochs (default: 10e2)  
        batches: int
            Number of batches (default: 10)  
        learningrate: float
            Starting learningrate (default: 10e-1)
        momentum: boolean
            Choose if ADAGRAD is perfomed with or without momentum (default: False)
        delta_momentum: float
            momentum parameter (default: 0.3)
        """
        
        X_train = self.X_train.copy()
        f_train = self.f_train.copy()
        beta = self.beta.copy()
        Giter = np.zeros((X_train.shape[1],X_train.shape[1]))
        delta  = 1e-8
        change = 0
        
        X_train = np.array_split(X_train,batches,axis=0)
        f_train = np.array_split(f_train,batches)
        
        np.random.seed(1999) #ensures reproducibility
        for itera in range(epochs):
            for i in range(batches):
                rd_ind = np.random.randint(batches)
                cost_fn = self.cost_fn
                gradient = cost_fn.grad(f_train[rd_ind],X_train[rd_ind],beta)
                Giter += gradient @ gradient.T
                coef = np.c_[learningrate/(delta+np.sqrt(np.diagonal(Giter)))]
                
                match momentum: 
                    case True: 
                        new_change = np.multiply(coef,gradient) + delta_momentum*change
                        change = new_change
                    case False: 
                        new_change = np.multiply(coef,gradient)
                    case _:
                        raise TypeError("<momentum> is a bolean variable")
                
                beta-=new_change
        
        return beta
    
    def rmsprop(self, epochs = int(10e2), batches = 10, learningrate= 10e-3, 
                t = 0.9):
        """
        Stochastic Gradient Decent with RMSprop
        
        Arguments
        ---------
        epochs: int
            Number of epochs (default: 10e2)  
        batches: int
            Number of batches (default: 10)  
        learningrate: float
            Starting learningrate (default: 10e-3)
        t: float
            averaging time of the second moment (default: 0.9)
        """
        
        X_train = self.X_train.copy()
        f_train = self.f_train.copy()
        beta = self.beta.copy()
        s = np.zeros((X_train.shape[1],1)) 
        delta  = 1e-8
        
        X_train = np.array_split(X_train,batches,axis=0)
        f_train = np.array_split(f_train,batches)
        np.random.seed(1999) #ensures reproducibility
        
        for itera in range(epochs):
            for i in range(batches):
                rd_ind = np.random.randint(batches)
                cost_fn = self.cost_fn
                gradient = cost_fn.grad(f_train[rd_ind],X_train[rd_ind],beta)
                s = t*s + (1-t)*np.power(gradient,2)
                coef = learningrate/np.sqrt(delta+np.sqrt(s))
                beta -= np.multiply(coef,gradient)
        
        return beta

    def adam(self, epochs = int(10e2), batches = 10, learningrate= 0.1, 
             t1 = 0.9, t2 = 0.99):
        """
        Stochastic Gradient Decent with ADAM
        
        Arguments
        ---------
        epochs: int
            Number of epochs (default: 10e2)  
        batches: int
            Number of batches (default: 10)    
        learningrate: float
            Starting learningrate (default: 10e-3)
        t1: float
            averaging time of the first moment (default: 0.9)
        t2: float
            averaging time of the second moment (defualt:0.99)
        """
        
        X_train = self.X_train.copy()
        f_train = self.f_train.copy()
        beta = self.beta.copy()
        m = np.zeros((X_train.shape[1],1))
        s = np.zeros((X_train.shape[1],1))
        delta  = 1e-8
        
        X_train = np.array_split(X_train,batches,axis=0)
        f_train = np.array_split(f_train,batches)
        
        np.random.seed(1999) #ensures reproducibility
        for itera in range(epochs):
            for i in range(batches):
                rd_ind = np.random.randint(batches)
                cost_fn = self.cost_fn
                gradient = cost_fn.grad(f_train[rd_ind],X_train[rd_ind],beta)
                m = t1 * m + (1-t1) * gradient
                m_hat = m / (1 - np.power(t1,itera+1))
                s = t2 * s + (1-t2) * np.power(gradient,2)
                s_hat = s / (1 - np.power(t2,itera+1))
                coef = learningrate/(delta+np.sqrt(s_hat))
                beta -= np.multiply(coef,m_hat)
        
        return beta
 
