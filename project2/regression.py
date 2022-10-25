#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""
class LinearRegression:
	""" Class for performing linear regression methods to fit 2D datasets with higher order polynomials"""


	def __init__(self, scale = False, ):
		""" Constructor for generating an instance of the class.

		Arguments
		---------
		order : int
			Highest order of polynomial model to fit data with.
		sigma : int or float
			Standard deviation parameter. (Default: 0.1)
		points : int
			Number of points for Franke function to test methods on. (Default: 20)
		X : array or list
			Ability to set a custom design matrix. (Default: lets scikit-learn's PolynomialFeatures subclass extract
													it using the max order and elements in the dataset we work on.)
		noise : bool
			Let's one turn on or off the noise applied to Franke function, follows th
			normal normal distribution N(0,1). (Default: True)
		scale : bool
			 Let's one turn on or of the scaling/centering of the data when needed by method used. (Default: True)
		data : str
			Path or filename of .tif dataset to work on. (Default: None and sets Franke function to work on)
		Raises
		------
		TypeError:
			If data is not a string indicating path to or name of .tif file is not specified correctly.
		"""

		def design_matrix(self, dataset, order):
			poly = PolynomialFeatures(degree=order)
			X = poly.fit_transform(], axis=-1), axis=0))
			return X

		def ols(self, X_train, z_train):
			beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
			return beta

		def ridge(self, X_train, z_train, hyperparam=0):
			hyperparam = self.hyperparam
			I = np.identity(X_train.T.shape[0])
			beta = np.linalg.pinv(X_train.T @ X_train + hyperparam*I) 
			beta = beta @ X_train.T @ z_train
			return beta

		def lasso(self, X_train, z_train, hyperparam=0):
			hyperparam = self.hyperparam
			lasso_regression = linear_model.Lasso(alpha=hyperparam, max_iter=int(1e6), tol=3e-2, fit_intercept=False)
			lasso_regression.fit(X_train, z_train)
			beta = lasso_regression.coef_
			return beta
 
