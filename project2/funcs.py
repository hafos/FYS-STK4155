#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import numpy as np
import scipy as sp
import itertools as itt
import matplotlib.pyplot as plt


class functions:
	def __init__(self, points = 20, sigma=0.1, dimension=1, order = 1,coef = None):
		""" Constructor for generating an instance of the class.

		Arguments
		---------
		points : int
			Number of points for Franke function to test methods on. (Default: 20)
		sigma : int or float
			Standard deviation parameter. (Default: 0.1)
		dimension:
			specifies the dimension of the polynomial generated. (Default: 1)
		order:
			specifies the order of the polynomial generated. (Default: 1)	
		coef : array 
			Defines all coeficients of the polynomial. (Default: all coeficients = 1)
			For multidimensional Polynomials the order of the Polynomials is given
			by the output of itertools.combinations_with_replacement
			e.g. for 2 dim up to order 2: x + y + z + x² + xy +xz + y² + yz + z² 
			
		------
		TypeError:
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
		
	 
func = functions(order = 2, coef = [0,1,0],sigma=0.1)
data, funcval = func.polynomial()
plt.scatter(data,funcval)