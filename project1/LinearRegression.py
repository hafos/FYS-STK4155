import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np 

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from scipy.special import comb

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.grid': True})
plt.rc('legend', frameon=False)

class LinearRegression:
	""" Class for performing linear regression methods to fit 2D datasets with higher order polynomials"""


	def __init__(self, order, sigma=0.1, points=20, X=None, noise=True, scale=False, data=None):
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

		self.order = order
		self.sigma = sigma
		self.points = points
		self.scale = scale

		# Datapoints
		np.random.seed(1999) # to ensure that we compare the different regression methods on the same datapoints
		if data == None:
			x, y = np.meshgrid(np.arange(0, 1, 1/points), np.arange(0, 1, 1/points))
			self.z = np.concatenate(self.franke_function(x, y, sigma, noise), axis=None)
			self.dataset = [x, y]
		elif data != None and isinstance(data, str):
			self.dataset = data # loads path to .tif dataset
		else:
			raise TypeError('Class keyword argument <data> needs to be path to .tif file in string format "PATH".')

		# Design Matrix
		if X == None:
			self.X = self.design_matrix(self.dataset, order)
		else:
			self.X = X

		# Score arrays
		self.MSE_train = np.zeros(order)
		self.MSE_test = np.zeros(order)
		self.R2_train = np.zeros(order)
		self.R2_test = np.zeros(order)
		self.BIAS = np.zeros(order)
		self.var = np.zeros(order, dtype=np.ndarray) # variance of beta parameters, confidence intervals sigma**2*(X.T)

		# number of terms given by the polynomial being complete homogenous symetric 
		n_terms = comb(len(self.dataset) + order, order, exact=True)
		self.beta = np.full((order, n_terms), np.nan)

	def franke_function(self, x, y, sigma, noise):
		""" Function for computing the Franke function we test the regression methods on """
		term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
		term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
		term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
		term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
		values = term1 + term2 + term3 + term4
		if noise == True:
			# values += sigma*np.random.rand(len(x),len(y))
			values += np.random.normal(0, sigma, (len(x),len(y))) # might want to soften noise?
			# values -= np.mean(values) # should we center?
		return values

	def design_matrix(self, dataset, order):
		poly = PolynomialFeatures(degree=order)
		X = poly.fit_transform(np.concatenate(np.stack([dataset[i] for i in range(0, len(dataset))], axis=-1), axis=0))
		return X

	def r2(self, func, func_tilde):
		mean = np.mean(func)
		R2 = -np.sum(np.power(func - func_tilde, 2))
		R2 /= np.sum(np.power(np.subtract(func, mean), 2))
		R2 += 1  
		return R2

	def scaling(self, data_to_scale):
		scaler = StandardScaler()
		scaler.fit(data_to_scale)
		data_scaled = scaler.transform(data_to_scale)
		return data_scaled

	# can get rid off
	def get_MSE(self, i, z, z_tilde):
		self.MSE_test[i-1] = np.mean(np.power(z - z_tilde, 2))
		self.MSE_train[i-1] = np.mean(np.power(z - z_tilde, 2))

	# def split(self, ratio=0.2): 
	# 	""" Function for splitting data into training and testing sets using scikit-learn """
	# 	X_train, X_test, z_train, z_test = train_test_split(self.X, self.z, test_size=ratio)#, random_state=1)
	#   if self.scale == True: 
			# X_test -= np.mean(X_train)
			# X_train -= np.mean(X_train)
			# z_scale = np.mean(z_train)
			# z_train -= z_scale
	# 	return X_train, X_test, z_train, z_test
	"""Remove ? """

	def ols(self, X_train, z_train):
		beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
		return beta
	
	def bootstrap(self, X_train, X_test, z_train, z_test, method=ols, n=100):
		""" Function for doing the bootstrap resampling method """
		z_scale = 0
		score = np.zeros(n)
		z_test_tilde = np.zeros((len(z_test), n))
		for j in range(0, n):
			X_train_resample, z_train_resample = resample(X_train, z_train, random_state = j)
			X_test_resample = X_test + 0
			# print(X_test)
			##rescaling
			if self.scale == True:
				# X_train_resample -= np.mean(X_train_resample, axis=0)
				X_train_resample = self.scaling(X_train_resample)
				# X_test_resample -= np.mean(X_test, axis=0)
				X_test_resample = self.scaling(X_test_resample)
				z_scale = np.mean(z_train_resample)         
				z_train_resample -= z_scale
				z_train_resample /= np.std(z_train_resample)
			beta = method(X_train_resample, z_train_resample)
			z_test_tilde[:, j] = X_test_resample @ beta  + z_scale
			# print(z_test_tilde[:, j])
			score[j] = np.mean((z_test - z_test_tilde[:, j])**2)
		return np.mean(score), z_test_tilde


	def execute_regression(self, method=ols, bootstrap=False, n=100):
		""" Method for doing the Ordinary Least Squares (OLS) regression """
		# if self.scale == True:
		# 	self.z = np.array_split(self.z, self.points)
		# 	self.z = self.scaling(self.z)
		# 	self.z = np.concatenate(self.z, axis=None)
		# 	self.X = self.scaling(self.X)
		X_train, X_test, z_train, z_test = train_test_split(self.X, self.z, test_size=0.2, random_state=1)
		z_scale = 0
		if self.scale == True:
			X_train = self.scaling(X_train)
			X_test = self.scaling(X_test)
			z_train -= np.mean(z_train)
			z_train /= np.std(z_train)
			z_test -= np.mean(z_test)
			z_test /= np.std(z_test)
			# self.scaling(z_train)
			# z_test = self.scaling(z_test)

		for i in range(1, self.order+1):
			# current number of terms in polynomial of order given as complete homogeneous symmetric
			current_n_terms = comb(len(self.dataset) + i, i, exact=True)
			#select only the terms of the full design matrix needed for the current order polynomial
			X_train_current = X_train[:,0:current_n_terms] + 0
			# X_train_current -= np.mean(X_train_current)
			X_test_current =  X_test[:,0:current_n_terms] + 0

			if bootstrap==True:
				z_train_tilde = np.zeros(len(z_train))
				#for fig 2.11 of Hastie, Tibshirani, and Friedman
				# if self.scale == True:
				# 	X_train_current = self.scaling(X_train_current)
				# 	z_scale = np.mean(z_train)
				beta = method(X_train_current, z_train - z_scale)
				z_train_tilde = X_train_current @ beta + z_scale

				X_train_current = X_train[:, 0:current_n_terms] + 0
				X_test_current =  X_test[:, 0:current_n_terms] + 0
				#Calcuate the errors
				self.MSE_test[i-1], z_test_tilde = self.bootstrap(X_train_current, X_test_current, z_train, z_test, method, n=n)
				self.MSE_train[i-1] = np.mean(np.power(z_train - z_train_tilde, 2))
				self.BIAS[i-1] = np.mean((z_test.reshape(-1, 1) - np.mean(z_test_tilde, axis=1, keepdims=True))**2 )
				self.var[i-1] = np.mean(np.var(z_test_tilde, axis=1, keepdims=True) )
				# print(self.MSE_test, i)
				# print(f"MSE - BIAS + var {(self.MSE_test[i-1] - (self.BIAS[i-1] + self.var[i-1]))}")
			
			else:
				
				#calc both errors and store the betas in the process
				# print(np.shape(np.linalg.pinv(X_train_current.T @ X_train) @ X_train_current.T @ z_train), np.shape(X_train_current.T), np.shape(z_train))
				self.beta[i-1][:current_n_terms] = method(X_train_current, z_train)
				z_train_tilde = X_train_current @ self.beta[i-1][~np.isnan(self.beta[i-1])] #+ z_scale
				z_test_tilde = X_test_current @ self.beta[i-1][~np.isnan(self.beta[i-1])]
				# self.get_MSE(i-1, )
				self.MSE_train[i-1] = np.mean(np.power(z_train - z_train_tilde, 2))
				self.MSE_test[i-1] = np.mean(np.power(z_test - z_test_tilde, 2)) #can delete these two if only need to call method
				self.R2_train[i-1] = self.r2(z_train, z_train_tilde)
				self.R2_test[i-1] = self.r2(z_test, z_test_tilde)
				# print(np.shape(z_train_tilde))
				# does not work... self.BIAS[i-1] = np.mean((z_test.reshape(-1, 1) - np.mean(z_test_tilde, axis=1, keepdims=True))**2 )
				# self.var[i-1] = np.var(z_trainz_test_tilde, axis=0, keepdims=True)
				# print(np.shape(np.var(z_train - z_train_tilde)))
				# print(type(np.var(z_train - z_train_tilde, keepdims=True)*np.diag(np.linalg.pinv(X_train_current.T @ X_train_current))))
				self.var[i-1] = np.var(z_train - z_train_tilde, keepdims=True)*np.diag(np.linalg.pinv(X_train_current.T @ X_train_current))
			
	def plot_franke_function(self):
		""" Method mainly for testing that dataset looks as expected """
		fig = plt.figure()
		ax = plt.axes(projection = '3d')
		z_plot = np.array_split(self.z, self.points)
		z_plot = np.array(z_plot) 
		surf = ax.plot_surface(self.dataset[0], self.dataset[1], z_plot, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
		# Customize the z axis.
		ax.set_zlim(-0.10, 1.40)
		# ax.zaxis.set_major_locator(LinearLocator(10))
		# ax.zaxis.set_major_formatter(FormatStrFormatter(’%.02f’))
		# Add a color bar which maps values to colors.
		ax.grid(False)
		fig.colorbar(surf, shrink=0.5, aspect=5)
		plt.show()

if __name__ == '__main__':
	""" Task b) """
	# LR_b = LinearRegression(order=5, scale=True, points=40)
	# # LR_b.plot_franke_function()
	# LR_b.execute_regression(method=LR_b.ols)
	# poly_degrees = np.arange(1, 6)
	# # # # print(var[0][0])
	# # # # print(np.shape(LR_b.var[-1][1][0]))
	# # # # print(len(betas[-1]))
	# plt.figure(figsize=(12,9))
	# plt.plot(poly_degrees, LR_b.MSE_test, label='MSE test')
	# plt.plot(poly_degrees, LR_b.MSE_train, label='MSE train')
	# plt.xlabel('Polynomial degree')
	# plt.ylabel('Mean Squared Error')
	# plt.figure(figsize=(12,9))
	# plt.plot(poly_degrees, LR_b.R2_test, label=r'$R^2 test$')
	# plt.plot(poly_degrees, LR_b.R2_train, label=r'$R^2 train$')
	# plt.xlabel('Polynomial degree')
	# plt.ylabel(r'$R^2 \; Score$')
	# plt.legend(); plt.show()

	# ### Beta coefficients
	# betas = LR_b.beta
	# var = LR_b.var[::-1]
	# plt.figure(figsize=(12, 9))
	# # # print(np.shape(LR_b.var))
	# # # print(LR_b.var[-1])
	# ax = plt.axes()
	# color = plt.cm.viridis(np.linspace(0.9, 0,11))
	# ax.set_prop_cycle(plt.cycler('color', color))#["axes.prop_cycle"] = plt.cycler('color', color)
	# ax.set_xticks([i for i in range(1, len(betas[-1])+1)])
	# for i, beta in enumerate(betas[::-1]):
	# 	coefficients = beta[~(np.isnan(beta))]
	# 	beta_indexes = np.arange(1, len(coefficients)+1)
	# 	plt.errorbar(beta_indexes, coefficients, yerr=np.sqrt(var[i]), marker='o', linestyle='--', capsize=2, label='d = %d' % (5-i))
	# """ We need to figure out what to do with the scaling.
	# 	Decide wether np.sqrt of variance as error bars or just the variance. 
	# 	  Also decide wether we have split's random_state be the same as random seed or set it 
	# 	  as something else as we have so far """
	# plt.xlabel('Order of polynomial')
	# plt.ylabel(r'$\beta\; coefficient \;value$')
	# plt.legend()
	# plt.show()

	""" Task c) """
	order = 12
	LR_c = LinearRegression(order=order, points=20, sigma=0.1, scale=True)
	# LR_c.plot_franke_function()
	# # print(LR_c.scale)
	LR_c.execute_regression(method=LR_c.ols, bootstrap=True, n=400)
	poly_degrees = np.arange(1, order+1)
	# print(LR_c.MSE_test)
	# print(LR_c.var)
	plt.plot(poly_degrees, LR_c.MSE_train, label='train')
	plt.plot(poly_degrees, LR_c.MSE_test, label='test')
	plt.legend()
	plt.show()
	plt.plot(poly_degrees, LR_c.BIAS, label='BIAS', color='orange')#, s=15)
	plt.plot(poly_degrees, LR_c.MSE_test, label='MSE_test', color='blue')#, s=15)
	plt.plot(poly_degrees, LR_c.var, label='var', color='red')#, s=15)     
	plt.legend()
	plt.show()

	""" Task d) """

