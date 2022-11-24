#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import numpy as np

def FrankeFunction(points = 100, sigma=0.25):
    """
    Generate data and function values for the Franke function
            
    Arguments
    
    ---------
    points: int
        Number of points for each dimensions. The total amount of datapoints is
        points squared (default: 100)
    sigma: float
        Normal distributed error added to the function values (default: 0.25)
    """
    
    np.random.seed(1999) 
    data = np.random.uniform(-1,1,(points,2))
    meshed_data = np.zeros((np.power(points,2),2))
    i = 0
    for variab in map(lambda x: np.reshape(x,(x.size)),np.meshgrid(*data.T)):
        meshed_data[:,i] = variab
        i+=1
    
    x = meshed_data[:,0]
    y = meshed_data[:,1]
    term1 =  0.75 * np.exp(-(0.25*(9*x-2)**2)      - 0.25*((9*y-2)**2))
    term2 =  0.75 * np.exp(-(     (9*x+1)**2)/49.0 -  0.1*(9*y+1))
    term3 =  0.5  * np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2  * np.exp(-(9*x-4)**2     -       (9*y-7)**2)
    np.random.seed(1999) 
    funcval = term1 + term2 + term3 + term4 + np.random.normal(0, sigma, meshed_data.shape[0])
    funcval = np.atleast_2d(funcval).reshape(-1,1)
    return meshed_data, funcval
