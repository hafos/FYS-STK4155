#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import numpy as np

class GradDecent:
    """Class for performing Gradient Decent methods on a given dataset"""
    def __init__(self, X_train = None, trainval = None, beta = None, 
                 costfunc = None):
        """ 
        Constructor for generating an instance of the class.
        
        Arguments
        
        ---------
        X_train: array
            train data values (default: None)
        trainval: array
            train function values (default: None)
        costfunc: func
            If one wants to use gradient decent, a cost function and the derivative
            has to be provided (default: None)
        learningrate: float
            Define the starting learning rate (default: 0.1)
        beta: float
            Define the starting beta (default is generated by np.random.rand)
       
        Errors
        ------
        TypeError:
            If no data is provided
            If no cost function is provided for gradient decent
        Index Error:
            Dimension of given starting beta is wrong
        """
        
        if X_train is None or trainval is None:
            raise TypeError("Class needs data as Input: <X_train> and <trainval> not provided")	
        if costfunc is None: 
            raise TypeError("Cost func is missing")
        if beta is None:             
            #ensures reproducibility
            np.random.seed(1999)
            beta = np.random.randn(X_train.shape[1],1)
        if beta.shape[0] != X_train.shape[1]: 
            raise IndexError("dim. of beta is wrong")
        
        self.X_train = X_train
        self.trainval = trainval
        self.costfunc = costfunc
        self.beta = beta
        

    def const(self, iterations = int(10e3), learningrate= 10e-3):
        """
        Gradient Decent with a constant learningrate
        
        Arguments
        ---------
        iterations: int
            Number of iterations (default: 10e3)   
        learningrate: float
            learningrate (default: 10e-3)
        """
        
        X_train = self.X_train
        trainval = self.trainval
        beta = self.beta.copy()
         
        for itera in range(int(iterations)):
            costfunc = self.costfunc
            gradient = costfunc.derivative(trainval,X_train,beta)
            beta -= learningrate*gradient
        return beta
    
    
    def adaptive(self, iterations = int(10e3), t_0= 10e-3, t_1 = 1.0):
        """
        Gradient Decent with a adaptive learningrate
        
        Arguments
        ---------
        iterations: int
            Number of iterations (default: 10e3)   
        learningrate: float
            learningrate (default: 10e-3)
        t_0: float
            parameter 1 (default: 10e-3)
        t_1: float
            parameter 2 (default: 1.0)
        """
        
        X_train = self.X_train
        trainval = self.trainval
        beta = self.beta.copy()
         
        for itera in range(int(iterations)):
            costfunc = self.costfunc
            gradient = costfunc.derivative(trainval,X_train,beta)
            learningrate = t_0/(t_1+itera)
            beta -= learningrate*gradient
        return beta
    
    def momentum(self, iterations = int(10e3), learningrate= 10e-3, 
                 delta_momentum = 0.1):
        
        """
        Momentum based Gradient Decent
        
        Arguments
        ---------
        iterations: int
            Number of iterations (default: 10e3)   
        learningrate: float
            Starting learningrate (default: 10e-0)
        delta_momentum: float
            momentum parameter (default: 0.1)
        """
        
        X_train = self.X_train
        trainval = self.trainval
        beta = self.beta.copy()
        change = 0
        
        for itera in range(iterations):
            costfunc = self.costfunc
            gradient = costfunc.derivative(trainval,X_train,beta)
            new_change = learningrate*gradient+delta_momentum*change
            beta -= new_change
            change = new_change
        return beta
    
    def adagrad(self, iterations = int(10e3), learningrate= 10e-0, 
                momentum = False, delta_momentum = 0.1):
        """
        Gradient Decent with ADAGRAD
        
        Arguments
        ---------
        iterations: int
            Number of iterations (default: 10e3)   
        learningrate: float
            Starting learningrate (default: 10e-3)
        momentum: boolean
            Choose if ADAGRAD is perfomed with or without momentum (default: False)
        delta_momentum: float
            momentum parameter (default: 0.1)
            
        Errors
        ---------
        TypeError: 
            if <momentum> is not bolean
        """
        
        X_train = self.X_train.copy()
        trainval = self.trainval.copy()
        beta = self.beta.copy()
        Giter = np.zeros((X_train.shape[1],X_train.shape[1]))
        delta  = 1e-8
        change = 0
        
        for itera in range(iterations):
            costfunc = self.costfunc
            gradient = costfunc.derivative(trainval,X_train,beta)
            Giter +=gradient @ gradient.T
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
    
    def rmsprop(self, iterations = int(10e3), learningrate= 10e-3, 
                t = 0.9):
        
        """
        Gradient Decent with RMSprop
        
        Arguments
        ---------
        iterations: int
            Number of iterations (default: 10e3)   
        learningrate: float
            Starting learningrate (default: 10e-3)
        t: float
            averaging time of the second moment (default: 0.9)
        """
        
        X_train = self.X_train
        trainval = self.trainval
        beta = self.beta.copy()
        s = np.zeros((X_train.shape[1],1))
        delta  = 1e-8
        
        for itera in range(0,iterations):
            costfunc = self.costfunc
            gradient = costfunc.derivative(trainval,X_train,beta)
            s = t*s + (1-t)*np.power(gradient,2)
            coef = learningrate/np.sqrt(delta+np.sqrt(s))
            beta -= np.multiply(coef,gradient)
        return beta
    
    def adam(self, iterations = int(10e3), learningrate= 10e-3, 
             t1 = 0.9, t2 = 0.99):
        """
        Gradient Decent with ADAM
        
        Arguments
        ---------
        iterations: int
            Number of iterations (default: 10e3)   
        learningrate: float
            Starting learningrate (default: 10e-3)
        t1: float
            averaging time of the first moment (default: 0.9)
        t2: float
            averaging time of the second moment (defualt:0.99)
        """
        
        X_train = self.X_train
        trainval = self.trainval
        beta = self.beta.copy()
        m = np.zeros((X_train.shape[1],1))
        s = np.zeros((X_train.shape[1],1))
        delta  = 1e-8
        
        for itera in range(0,iterations):
            costfunc = self.costfunc
            gradient = costfunc.derivative(trainval,X_train,beta)
            m = t1 * m + (1-t1) * gradient
            m_hat = m / (1 - np.power(t1,itera+1))
            s = t2 * s + (1-t2) * np.power(gradient,2)
            s_hat = s / (1 - np.power(t2,itera+1))
            coef = learningrate/(delta+np.sqrt(s_hat))
            beta -= np.multiply(coef,m_hat)
        return beta