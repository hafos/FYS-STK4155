#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import numpy as np
import scipy as sp
import itertools as itt


class functions:
    def __init__(self, points = 20, sigma=0.1, dimension=1, order = 1,coef = None):
        """ Constructor for generating an instance of the class.
        
        Arguments
		---------
		points : int
			Number of points for Franke function to test methods on. (Default: 20)
		sigma : int or float
			Standard deviation parameter. (Default: 0.1)
		dimension: int
			specifies the dimension of the polynomial generated. (Default: 1)
		order: int
			specifies the order of the polynomial generated. (Default: 1)	
		coef : array 
			Defines all coeficients of the polynomial. (Default: all coeficients = 1)
			For multidimensional Polynomials the order of the Polynomials is given
			by the output of itertools.combinations_with_replacement
			e.g. for 2 dim up to order 2: x + y + z + x² + xy +xz + y² + yz + z² 
			
        Errors
		------
		IndexError:
			If number of coef doesnt mathch dimensions & order
		"""
        
        numberofterms = sp.special.comb(dimension + order,order,exact=True)
        if coef == None: 
            coef = np.ones(numberofterms)
        if numberofterms!=len(coef):
            raise IndexError(f'expected {sp.special.comb(dimension + order,order,exact=True)} coeficients')
        
        self.points = points
        self.sigma = sigma
        self.dimension = dimension
        self.order = order
        self.coef = coef
        
        #ensures reproducibility
        np.random.seed(1999) 
        self.data = np.random.uniform(-1,1,(points,dimension))
        meshed_data = np.zeros((np.power(points,dimension),dimension))
        i = 0
        for variab in map(lambda x: np.reshape(x,(x.size)),np.meshgrid(*self.data.T)):
            meshed_data[:,i] = variab
            i+=1
        self.data = meshed_data
		
    def polynomial(self):
		#Calc indices for multidim. polynomial up to the given order
        powers = [x for i in range(1,self.order+1) for x in itt.combinations_with_replacement(range(self.dimension), i)]
        #Create polynomial
        polynom = np.ones((self.data.shape[0],len(powers)+1))
        polynom[:] *= self.coef[:]
        for j in range(len(powers)):
            for i in powers[j]:
                if polynom[:,j+1].all == 0: break
                polynom[:,j+1] *= self.data[:,i]  
        np.random.seed(1999) 
        funcval = np.sum(polynom,axis=1,keepdims=True) + np.random.normal(0, self.sigma, (self.data.shape[0],1))
        return self.data, funcval
    
    def FrankeFunction(self):
        if self.data.shape[1] !=2:
            raise IndexError('the Franke Function is two dimensional')
        x = self.data[:,0]
        y = self.data[:,1]
        term1 =  0.75 * np.exp(-(0.25*(9*x-2)**2)      - 0.25*((9*y-2)**2))
        term2 =  0.75 * np.exp(-(     (9*x+1)**2)/49.0 -  0.1*(9*y+1))
        term3 =  0.5  * np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2  * np.exp(-(9*x-4)**2     -       (9*y-7)**2)
        np.random.seed(1999) 
        funcval = term1 + term2 + term3 + term4 + np.random.normal(0, self.sigma, self.data.shape[0])
        funcval = np.atleast_2d(funcval).T
        return self.data, funcval