#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""
import time
import numpy as np
import scipy as sp
import itertools as itt
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from regression import LinearRegression,GradDecent,StochGradDecent

def CostOLS(y,X,beta):
    return 1/X.shape[0]*((y-X@beta).T@(y-X@beta))

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
        if coef == None: coef = np.ones(numberofterms)
        if numberofterms!=len(coef):
            raise IndexError(f'expected {sp.special.comb(dimension + order,order,exact=True)} coeficients')
        self.points = points
        self.sigma = sigma
        self.dimension = dimension
        self.order = order
        self.coef = coef
        #ensures reproducibility
        np.random.seed(1999)
        #generates random data points
        #self.data = np.random.rand(points,dimension)  
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
        funcval = np.sum(polynom,axis=1,keepdims=True) + np.random.normal(0, self.sigma, (self.data.shape[0],1))
        return self.data, funcval
	
    def design_matrix(self):
        poly = PolynomialFeatures(degree=self.order)
        X = poly.fit_transform(self.data)
        return X
	
	 
func = functions(order = 3, dimension=1, sigma=0,points=100)
data, funcval = func.polynomial()
poly = PolynomialFeatures(degree=3)
X = poly.fit_transform(data)

reg = LinearRegression(X,funcval)
beta = reg.ols()

gd = GradDecent(X,funcval,CostOLS)
start = time.time()
beta2 = gd.const(iterations=200)
end = time.time()
print(end - start)
beta3 = gd.momentum(iterations=500)

start = time.time()
sd = StochGradDecent(X,funcval,CostOLS)
beta4 = sd.const(iterations=200)
end = time.time()
print(end - start)
beta5 = sd.momentum(iterations=200)
#print(beta)
print(beta2)
print(beta4)
print(beta5)
plt.scatter(data,funcval)
