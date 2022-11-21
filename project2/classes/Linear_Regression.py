#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import numpy as np
from sklearn import linear_model

class LinearRegression:
    """Class for performing linear regression methods on a given dataset"""
    def __init__(self, X_train = None, trainval = None):
        """ 
        Constructor for generating an instance of the class.
        
        Arguments
        ---------
        X_train: array
            train data values (default = None)
        trainval: array
            train function values (default = None)
        
        Errors
        ------
        TypeError:
            If no data or order is provided
        """
        
        if X_train is None or trainval is None:
            raise TypeError("Class needs data as Input: <X_train> and <trainval> not provided")	
        self.X_train = X_train
        self.trainval = trainval
        
    def ols(self):
        """
        Function to compute ideal parameters with ordinary least square
        """
        
        X_train = self.X_train
        trainval = self.trainval
        
        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ trainval
        return beta
    
    def ridge(self, hyperparam=0.0):
        """
        Function to compute ideal parameters with ridge regression
        
        Arguments
        ---------
        hyperparam: float
            Rige Parameter (default: 0.0)
        """
        
        X_train = self.X_train
        trainval = self.trainval
        
        I = np.identity(X_train.T.shape[0])
        beta = np.linalg.pinv(X_train.T @ X_train + hyperparam*I) 
        beta = beta @ X_train.T @ trainval
        return beta
    
    def lasso(self, hyperparam=0):
        """
        Function to compute ideal parameters with lasso regression
        Arguments
        ---------
        hyperparam: float
            Lasso Parameter (default: 0.0)
        """
        
        X_train = self.X_train
        trainval = self.trainval
        
        lasso_regression = linear_model.Lasso(alpha=hyperparam, max_iter=int(1e6), tol=3e-2, fit_intercept=False)
        lasso_regression.fit(X_train, trainval)
        beta = lasso_regression.coef_
        return beta 
