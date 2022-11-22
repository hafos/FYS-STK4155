"""
Date: 21.11.22 

@author: semyaat
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import time

import sys
sys.path.append('src')

from Linear_Regression import LinearRegression
from Grad_Decent import GradDecent
from S_Grad_Decent import StochGradDecent
from gen_data import functions
from cost_act_func import CostOLS_beta, Ridge_beta


## Initialize 
func = functions(dimension=2, sigma=0.25 , points=100)
costfunc = CostOLS_beta

data, funcval = func.FrankeFunction()


# def plot_franke_function(self):
# 		""" Method mainly for testing that dataset looks as expected """
# 		fig = plt.figure()
# 		ax = plt.axes(projection = '3d')
# 		z_plot = np.array_split(self.z, self.points)
# 		z_plot = np.array(z_plot) 
# 		surf = ax.plot_surface(self.dataset[0], self.dataset[1], z_plot, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
# 		# Customize the z axis.
# 		ax.set_zlim(-0.10, 1.40)
# 		ax.grid(False)
# 		fig.colorbar(surf, shrink=0.5, aspect=5)
# 		plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

# def FrankeFunction(x,y):
#     term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
#     term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
#     term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
#     term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
#     return term1 + term2 + term3 + term4

# z = FrankeFunction(x, y)

x = data[0]
y = data[1]
z = funcval 
# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()