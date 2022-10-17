from audioop import cross
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
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
from imageio import imread

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.grid': True})
plt.rc('legend', frameon=False)
params = {'legend.fontsize': 25,
			'figure.figsize': (12, 9),
			'axes.labelsize': 25,
			'axes.titlesize': 25,
			'xtick.labelsize': 'x-large',
			'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)

class LinearRegression:
	""" Class for performing linear regression methods to fit 2D datasets with higher order polynomials"""


	def __init__(self, order, sigma=0.1, points=20, X=None, noise=True, scale=False, data=None, reduce_factor=0, x_pos=0, y_pos=0):
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
			self.data = imread(data) # loads path to .tif dataset
			self.z = self.data[y_pos:y_pos+300, x_pos:x_pos+300] # symetric slice of data with varried topography
			if reduce_factor != 0:
				self.z = self.z[::reduce_factor, ::reduce_factor]
			self.z = self.z - np.mean(self.z) # centering dataset to more closely reflect reality
			x, y = np.meshgrid(np.arange(0, 1, 1/len(self.z[0])), np.arange(0, 1, 1/len(self.z[1])))
			self.dataset = [x, y]
			self.z = np.concatenate(self.z, axis=None)
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

	def ridge(self, X_train, z_train, hyperparam=0):
		if hasattr(self, 'hyperparam'):
			hyperparam = self.hyperparam
		I = np.identity(X_train.T.shape[0])
		beta = np.linalg.pinv(X_train.T @ X_train + hyperparam*I) 
		beta = beta @ X_train.T @ z_train
		return beta

	def lasso(self, X_train, z_train, hyperparam=0):
		if hasattr(self, 'hyperparam'):
			hyperparam = self.hyperparam
		# change tol when scaling fixed so that converges
		lasso_regression = linear_model.Lasso(alpha=hyperparam, max_iter=int(1e6), tol=3e-2, fit_intercept=False)
		lasso_regression.fit(X_train, z_train)
		beta = lasso_regression.coef_
		return beta
	
	def bootstrap(self, X_train, X_test, z_train, z_test, method=ols, n=100):
		""" Function for doing the bootstrap resampling method """
		z_scale = 0
		score = np.zeros(n)
		z_test_tilde = np.zeros((len(z_test), n))
		for i in range(n):
			X_train_resample, z_train_resample = resample(X_train, z_train, random_state = i)
			X_test_resample = X_test + 0
			##rescaling
			# if self.scale == True:
			# 	# X_train_resample -= np.mean(X_train_resample, axis=0)
			# 	X_train_resample = self.scaling(X_train_resample)
			# 	# X_test_resample -= np.mean(X_test, axis=0)
			# 	X_test_resample = self.scaling(X_test_resample)
			# 	z_scale = np.mean(z_train_resample)         
			# 	z_train_resample -= z_scale
			# 	z_train_resample /= np.std(z_train_resample)
			beta = method(X_train_resample, z_train_resample)
			z_test_tilde[:, i] = X_test_resample @ beta  + z_scale
			score[i] = np.mean((z_test - z_test_tilde[:, i])**2)
		return np.mean(score), z_test_tilde

	def crossval(self, X, z, method, folds):
		z_scale = 0
		score = np.zeros(folds)
		rs = KFold(n_splits=folds, shuffle=True, random_state=1)
		i = 0
		for train_index, test_index in rs.split(X):
			X_train = X[train_index] + 0
			z_train = z[train_index] + 0
			X_test = X[test_index] + 0
			z_test = z[test_index]
			# if self.scale == True:
			# 	X_test  = self.scaling(X_test) # check if should be train similar in bootstrap np.mean(A_train,axis=0)
			# 	X_train = self.scaling(X_train) #np.mean(A_train,axis=0)
			# 	z_scale = np.mean(z_train)
			# 	z_train -= z_scale
			# 	z_train /= np.std(z_train)
			beta = method(X_train, z_train)
			z_test_tilde = X_test @ beta + z_scale
			score[i] = np.sum((z_test_tilde - z_test)**2)/np.size(z_test_tilde)
			i +=1        
		return np.mean(score)


	def execute_regression(self, method=ols, bootstrap=False, n=100, crossval=False, kfolds=10, hyperparams=0):
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
		if crossval == True and hyperparams == 0:
			self.MSE_CV = np.zeros((len(kfolds), self.order))
		elif crossval == True and hyperparams != 0:
			self.MSE_crossval = np.zeros((len(hyperparams), self.order))
		if bootstrap == True and hyperparams != 0:
			self.MSE_bootstrap = np.zeros((len(hyperparams), self.order))
			self.BIAS_bootstrap = np.zeros((len(hyperparams), self.order)) # For BVT analysis
			self.var_bootstrap = np.zeros((len(hyperparams), self.order))  # For BVT analysis
			self.hyperparam = hyperparams[0]

		for i in range(1, self.order+1):
			print(f'Computing for order: {i}' ) # for testing purposes and to see how far code
			# current number of terms in polynomial of order given as complete homogeneous symmetric
			current_n_terms = comb(len(self.dataset) + i, i, exact=True)
			#select only the terms of the full design matrix needed for the current order polynomial
			X_train_current = X_train[:,0:current_n_terms] + 0
			# X_train_current -= np.mean(X_train_current)
			X_test_current =  X_test[:,0:current_n_terms] + 0

			if bootstrap == True:
				z_train_tilde = np.zeros(len(z_train))
				#for fig 2.11 of Hastie, Tibshirani, and Friedman
				# if self.scale == True:
				# 	X_train_current = self.scaling(X_train_current)
				# 	z_scale = np.mean(z_train)
				# beta = method(X_train_current, z_train - z_scale)
				# z_train_tilde = X_train_current @ beta + z_scale

				X_train_current = X_train[:, 0:current_n_terms] + 0
				X_test_current =  X_test[:, 0:current_n_terms] + 0
				#Calcuate the errors
				if bootstrap == True and hyperparams == 0:
					beta = method(X_train_current, z_train - z_scale)
					z_train_tilde = X_train_current @ beta + z_scale
					self.MSE_test[i-1], z_test_tilde = self.bootstrap(X_train_current, X_test_current, z_train, z_test, method, n=n)
					self.MSE_train[i-1] = np.mean(np.power(z_train - z_train_tilde, 2))
					self.BIAS[i-1] = np.mean((z_test.reshape(-1, 1) - np.mean(z_test_tilde, axis=1, keepdims=True))**2 )
					self.var[i-1] = np.mean(np.var(z_test_tilde, axis=1, keepdims=True) )
				elif bootstrap == True and hyperparams != 0:
					k = 0
					for hyperparam in hyperparams:
						self.hyperparam = hyperparam
						self.MSE_bootstrap[k, i-1], z_test_tilde = self.bootstrap(X_train_current, X_test_current, z_train, z_test, method, n=n)
						self.BIAS_bootstrap[k, i-1] = np.mean((z_test.reshape(-1, 1) - np.mean(z_test_tilde, axis=1, keepdims=True))**2 )
						self.var_bootstrap[k, i-1] = np.mean(np.var(z_test_tilde, axis=1, keepdims=True) )
						k += 1
						# print(f"MSE - BIAS + var {(self.MSE_test[i-1] - (self.BIAS[i-1] + self.var[i-1]))}")
			elif crossval == True:
				X_curr = self.X[:, 0:current_n_terms] + 0
				if crossval == True and hyperparams == 0:
					k = 0
					for folds in kfolds: 
						self.MSE_CV[k, i-1] = self.crossval(X_curr, self.z, method, folds)
						k += 1
				elif crossval == True and hyperparams != 0:
					k = 0
					for hyperparam in hyperparams:
						self.hyperparam = hyperparam
						self.MSE_crossval[k, i-1] = self.crossval(X_curr, self.z, method, folds=kfolds)
						k += 1
					
			else:
				#calc both errors and store the betas in the process
				self.beta[i-1][:current_n_terms] = method(X_train_current, z_train)
				z_train_tilde = X_train_current @ self.beta[i-1][~np.isnan(self.beta[i-1])] #+ z_scale
				z_test_tilde = X_test_current @ self.beta[i-1][~np.isnan(self.beta[i-1])]
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

	def plot_terrain(self):
		""" Plot entire terrain dataset """
		fig, ax = plt.subplots()
		# fig = plt.figure()
		# ax = plt.axes(projection = '3d')
		plt.title('Terrain')
		ax.imshow(self.data, cmap='viridis')
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.show()
	
	def plot_terrain_3D(self):
		""" Plot 3D terrain of zoomed in area """
		fig = plt.figure()
		ax = plt.axes(projection = '3d')
		plt.title('Terrain 3D')
		z_plot = np.array_split(self.z, len(self.dataset[0]))
		z_plot = np.array(z_plot) 
		surf = ax.plot_surface(self.dataset[0], self.dataset[1], z_plot, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
		fig.colorbar(surf, shrink=0.5, aspect=5)
		plt.show()


if __name__ == '__main__':
	""" Task b) """
	# LR_b = LinearRegression(order=5, scale=True, points=40)
	# # LR_b.plot_franke_function()
	# LR_b.execute_regression(method=LR_b.ols)
	# poly_degrees = np.arange(1, 6)
	# fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12,9))
	# ax[0].plot(poly_degrees, LR_b.MSE_test,  label='MSE test',  color='orange', linestyle='--')
	# ax[0].plot(poly_degrees, LR_b.MSE_train, label='MSE train', color='orange')
	# ax[0].legend()
	# ax[0].set(ylabel="MSE score")
	# ax[1].plot(poly_degrees, LR_b.R2_test,   label=r'R$^2$ test', color='b', linestyle='--')
	# ax[1].plot(poly_degrees, LR_b.R2_train,  label=r'R$^2$ train', color='b')
	# ax[1].set(ylabel=r"R$^2$ score")
	# ax[1].legend()
	# plt.xlabel('Polynomial degree')
	# fig.tight_layout()
	# plt.show()
	# plt.savefig("figures/FrankeFunction/OLS_scores.pdf")

	# ### Beta coefficients
	# betas = LR_b.beta
	# var = LR_b.var[::-1]
	# ax = plt.axes()
	# color = plt.cm.viridis(np.linspace(0.9, 0,11))
	# ax.set_prop_cycle(plt.cycler('color', color))
	# ax.set_xticks([i for i in range(1, len(betas[-1])+1)])
	# for i, beta in enumerate(betas[::-1]):
	# 	coefficients = beta[~(np.isnan(beta))]
	# 	beta_indexes = np.arange(1, len(coefficients)+1)
	# 	plt.errorbar(beta_indexes, coefficients, yerr=np.sqrt(var[i]), marker='o', linestyle='--', capsize=4, label='d = %d' % (5-i))
	# plt.xlabel(r'$\beta$ coefficient number')
	# plt.ylabel(r'$\beta$ coefficient value')
	# plt.legend()
	# plt.tight_layout()
	# plt.show()
	# plt.savefig("figures/FrankeFunction/OLS_beta.pdf")

	""" Task c) """
	# order = 12
	# LR_c = LinearRegression(order=order, points=20, sigma=0.1, scale=True)
	# LR_c.execute_regression(method=LR_c.ols, bootstrap=True, n=300)
	# poly_degrees = np.arange(1, order+1)
	# plt.plot(poly_degrees, LR_c.MSE_train, label='train', color='orange', linestyle='--')
	# plt.plot(poly_degrees, LR_c.MSE_test,  label='test',  color='orange')
	# plt.legend()
	# plt.xlabel("Polynomial degree")
	# plt.ylabel("MSE score")
	# plt.tight_layout()
	# plt.show()
	# plt.savefig("figures/FrankeFunction/OLS_bootstrap.pdf")

	# plt.plot(poly_degrees, LR_c.BIAS,     label=r'BIAS$^2$',     color='red')
	# plt.plot(poly_degrees, LR_c.MSE_test, label='MSE test', color='orange')
	# plt.plot(poly_degrees, LR_c.var,      label='var',      color='green')   
	# plt.legend()
	# plt.xlabel("Polynomial degree")
	# plt.ylabel("score")
	# plt.tight_layout() 
	# plt.show()
	# plt.savefig("figures/FrankeFunction/OLS_biasvar.pdf")

	""" Task d) """
	""" Fails when scale == True, need to fix scaling, remove from crossval func?
		Add cross_val_score from sklearn to compare against?"""
	# order = 12
	# LR_d = LinearRegression(order=order, points=20, scale=False)
	# kfolds = [i for i in range(5, 11)]
	# LR_d.execute_regression(method=LR_d.ols, bootstrap=True, n=300)
	# LR_d.execute_regression(method=LR_d.ols, crossval=True, kfolds=kfolds)
	# poly_degrees = np.arange(1, order+1)
	# fig, ax = plt.subplots()
	# plt.plot(poly_degrees, LR_d.MSE_train, label='bootstrap train', color='k', linestyle='--')
	# plt.plot(poly_degrees, LR_d.MSE_test,  label='bootstrap test', color='k')
	# color = plt.cm.cool(np.linspace(0.9, 0,11))
	# ax.set_prop_cycle(plt.cycler('color', color))
	# for k in range(len(kfolds)):
	# 	plt.plot(poly_degrees, LR_d.MSE_CV[k], label=f'k = {kfolds[k]}')
	# plt.legend(loc='upper center')
	# plt.xlabel("Polynomial degree")
	# plt.ylabel("MSE score")
	# plt.tight_layout()
	# plt.savefig("figures/FrankeFunction/OLS_crossval.pdf")
	# plt.show()
	

	""" Task e) """
	""" Ridge Bootstrap """

	order = 12
	poly_degrees = np.arange(1, order+1)
	hyperparams = [10**i for i in range(-10, 0)]
	extent = [poly_degrees[0], poly_degrees[-1], np.log10(hyperparams[0]), np.log10(hyperparams[-1])]
	LR_e = LinearRegression(order=order, points=20, scale=True)
	LR_e.execute_regression(method=LR_e.ridge, bootstrap=True, n=100, hyperparams=hyperparams)
	MSE_ridge_bootstrap = LR_e.MSE_bootstrap
	# print(np.shape(MSE_ridge_bootstrap))
	min_MSE_idx = divmod(MSE_ridge_bootstrap.argmin(), MSE_ridge_bootstrap.shape[1])
	min_MSE_idx = divmod(MSE_ridge_bootstrap.argmin(), MSE_ridge_bootstrap.shape[0])
	# print(min_MSE_idx)

	""" Ridge heatmap """
	# print(MSE_ridge_bootstrap.min())
	# ymin, xmin = MSE_ridge_bootstrap) == MSE_ridge_bootstrap)
	# print(np.shape(MSE_ridge_bootstrap), xmin, ymin)
	# i, j = np.unravel_index(np.argmin(MSE_ridge_bootstrap), np.shape(MSE_ridge_bootstrap))
	fig, ax = plt.subplots()
	plt.contourf(MSE_ridge_bootstrap, extent=extent, levels=30)#(order*len(hyperparams)))
	# plt.contourf(MSE_ridge_bootstrap, extent=extent, levels=30)#(order*len(hyperparams)))
	# plt.plot(poly_degrees[j], hyperparams[i], 'o')
	# ax.set_xticklabels(range(1, order+1))
	# # ax.set_yticklabels(np.log10(hyperparams),out=np.zeros_like(hyperparams), where=(hyperparams!=0))
	# # plt.contourf(poly_degrees, hyperparams, MSE_ridge_bootstrap, cmap=plt.cm.magma, levels=30)
	# # plt.plot(min_MSE_idx[0], min_MSE_idx[1], 'o', color='red')
	plt.xlabel("Polynomial degree")
	plt.ylabel(r"Penalty parameter [log$_{10}$]")
	cbar = plt.colorbar(pad=0.01)
	cbar.set_label('MSE score')
	plt.tight_layout()
	plt.savefig("figures/FrankeFunction/Ridge_bootstrap.pdf")
	plt.show()
	# # sns.heatmap(MSE_ridge_bootstrap, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'},fmt='.1e')

	# # # ax.set_title("Test Accuracy BSE 2")
	# # # ax.set_ylabel("order")
	# # # ax.set_xlabel("log$_{10}(\lambda)$")
	# # ax.add_patch(plt.Rectangle((min_MSE_idx[0], min_MSE_idx[1]), 1, 1, fc='none', ec='red', lw=2, clip_on=False))
	# # plt.show()
	# # plt.plot(poly_degrees, LR_e.MSE_train, label='train')
	# # plt.plot(poly_degrees, LR_e.MSE_test, label='test')
	# # plt.legend()
	# # plt.show()

	""" Ridge bias-var analysis analysis with bootstrap """
	# BIAS_ridge_bootstrap = LR_e.BIAS_bootstrap
	# var_ridge_bootstrap = LR_e.var_bootstrap
	# for k in range(len(hyperparams)):
	# 	h1 = plt.plot(poly_degrees, MSE_ridge_bootstrap[k],  label='MSE test',  color='orange', alpha=k*0.1)#, s=15)
	# 	h2 = plt.plot(poly_degrees, BIAS_ridge_bootstrap[k], label=r'BIAS$^2$', color='blue',   alpha=k*0.1)#, s=15)
	# 	h3 = plt.plot(poly_degrees, var_ridge_bootstrap[k],  label='var',       color='red',    alpha=k*0.1)#, s=15)
	# 	plt.legend(handles=[h1[0], h2[0], h3[0]])#labels=["MSE_test", "BIAS", "Variance"])
	# plt.xlabel("Polynomial degree")
	# plt.ylabel("MSE score")
	# plt.tight_layout()
	# plt.savefig("figures/FrankeFunction/Ridge_biasvar.pdf")
	# plt.show()

	""" Ridge Cross validation heatmap """
	# kfolds = [i for i in range(5, 11)]
	# LR_e.execute_regression(method=LR_e.ridge, crossval=True, kfolds=10, hyperparams=hyperparams)
	# MSE_ridge_crossval = LR_e.MSE_crossval
	# # min_MSE_idx = divmod(MSE_ridge_crossval.argmin(), MSE_ridge_crossval.shape[1])
	# fig, ax = plt.subplots()
	# plt.contourf(MSE_ridge_crossval, extent=extent, levels=30)
	# # sns.heatmap(MSE_ridge_crossval.T, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'},fmt='.1e')
	# # ax.add_patch(plt.Rectangle((min_MSE_idx[0], min_MSE_idx[1]), 1, 1, fc='none', ec='red', lw=2, clip_on=False))
	# plt.xlabel("Polynomial degree")
	# plt.ylabel(r"Penalty parameter [log$_{10}$]")
	# cbar = plt.colorbar(pad=0.01)
	# cbar.set_label('MSE score')
	# plt.tight_layout()
	# plt.savefig("figures/FrankeFunction/Ridge_crossval.pdf")
	# plt.show()

	""" Task f) """
	""" Lasso Bootstrap """
	""" Lasso does not converge when scale=True, hints that scaling is not implemented correctly """

	order = 20
	poly_degrees = np.arange(1, order+1)
	hyperparams = [10**i for i in range(-10, 0)]
	extent = [poly_degrees[0], poly_degrees[-1], np.log10(hyperparams[0]), np.log10(hyperparams[-1])]
	LR_f = LinearRegression(order=order, points=20, scale=False)

	# LR_f.execute_regression(method=LR_f.lasso, bootstrap=True, n=10, hyperparams=hyperparams)
	# MSE_lasso_bootstrap = LR_f.MSE_bootstrap
	# print(np.shape(MSE_ridge_bootstrap))
	# min_MSE_idx = divmod(MSE_ridge_bootstrap.argmin(), MSE_ridge_bootstrap.shape[1])
	""" Lasso heatmap boot """
	# fig, ax = plt.subplots()
	# plt.contourf(MSE_lasso_bootstrap, extent=extent, levels=30)#(order*len(hyperparams)))
	# # plt.contourf(poly_degrees, hyperparams, MSE_ridge_bootstrap, cmap=plt.cm.magma, levels=30)
	# # plt.plot(min_MSE_idx[0], min_MSE_idx[1], 'o', color='red')
	# plt.xlabel("Polynomial degree")
	# plt.ylabel(r"Penalty parameter [log$_{10}$]")
	# cbar = plt.colorbar(pad=0.01)
	# cbar.set_label('MSE score')
	# plt.tight_layout()
	# plt.savefig("figures/FrankeFunction/Lasso_bootstrap.pdf")
	# plt.show()

	""" Lasso heatmap Cross Validation"""
	# kfolds = [i for i in range(5, 11)]
	# LR_f.execute_regression(method=LR_f.lasso, crossval=True, kfolds=10, hyperparams=hyperparams)
	# MSE_lasso_crossval = LR_f.MSE_crossval
	# fig, ax = plt.subplots()
	# plt.contourf(MSE_lasso_crossval, extent=extent, levels=30)
	# # sns.heatmap(MSE_lasso_crossval.T, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'},fmt='.1e')
	# # ax.add_patch(plt.Rectangle((min_MSE_idx[0], min_MSE_idx[1]), 1, 1, fc='none', ec='red', lw=2, clip_on=False))
	# plt.xlabel("Polynomial degree")
	# plt.ylabel(r"Penalty parameter [log$_{10}$]")
	# cbar = plt.colorbar(pad=0.01)
	# cbar.set_label('MSE score')
	# plt.tight_layout()
	# plt.savefig("figures/FrankeFunction/Lasso_crossval.pdf")
	# plt.show()
	
