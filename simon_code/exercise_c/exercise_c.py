import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed 

#import all needed functions located in seperate documents
from FrankeFunction import FrankeFunction
from functions import *

np.random.seed(2018)
noise = True
n_bootstrap = 100

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
sigma = 0.05
x, y = np.meshgrid(x,y)
z = np.concatenate(FrankeFunction(x, y),axis=None)
if noise == True: z += sigma*np.random.randn(len(z))

##Approximation
max_order = 14
#bring variables in the right form
variables=[x,y]
#the dimension of the array needed is given Complete homogeneous symmetric polynmial
numbofterms = sp.special.comb(len(variables) + max_order,max_order,exact=True)
beta = np.zeros((max_order,numbofterms))

#Error arrays
MSE_train = np.zeros(max_order)
MSE_test = np.zeros(max_order)
BIAS = np.zeros(max_order)
var = np.zeros(max_order)

##Full designe Matrix
A = DM(variables,max_order)
#split data into test & train
A_train, A_test, z_train, z_test = train_test_split(A, z, test_size=0.2)
#Train i do need need for the second part of the exercise
f_approx_test = np.empty((len(z_test), n_bootstrap+1))


for i in range(1,max_order+1):
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)
    ATrCur = A_train[:,0:currentnot]
    ATeCur = A_test[:,0:currentnot]
    beta[i-1][:currentnot] = np.linalg.inv(ATrCur.T @ ATrCur) @ ATrCur.T @ z_train
    f_approx_test[:,0] = ATeCur @ beta[i-1][beta[i-1] != 0]
    f_approx_train = ATrCur @ beta[i-1][beta[i-1] != 0]
    MSE_train[i-1] = 1/len(z_train)*np.sum(np.power(z_train-f_approx_train,2))
    for j in range(n_bootstrap+1):
        A_res, z_res = resample(ATrCur, z_train)
        beta[i-1][:currentnot] = np.linalg.pinv(A_res.T @ A_res) @ A_res.T @ z_res
        f_approx_test[:,j] = ATeCur @ beta[i-1][beta[i-1] != 0]
    MSE_test[i-1] = np.mean( np.mean((z_test.reshape(-1, 1) - f_approx_test)**2, axis=1, keepdims=True) )
    #Calc BIAS & variance
    BIAS[i-1] = np.mean( (z_test.reshape(-1, 1) - np.mean(f_approx_test, axis=1, keepdims=True))**2 )
    var[i-1] = np.mean( np.var(f_approx_test, axis=1, keepdims=True) )
    print(MSE_test[i-1]-(BIAS[i-1]+var[i-1]))

##reproduce fig 2.11 of Hastie, Tibshirani, and Friedman
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_train, label='train', color='orange', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_test, label='test', color='blue', s=15)
plt.legend(loc = "upper center")
plt.show()

#test= var+BIAS+np.power(sigma,2)

plt.scatter(np.arange(1, max_order+1, 1.0), BIAS, label='BIAS', color='orange', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_test, label='test', color='blue', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), var, label='var', color='red', s=15)     
plt.legend()
plt.show()



