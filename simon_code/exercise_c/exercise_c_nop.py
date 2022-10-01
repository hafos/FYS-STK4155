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
scaling = True
noise = True
sigma = 0.05
#Maximum number of bootstraps
n_bootstrap = 100
b_stepsize = 10
#Maximum number of data points
datapoints = 100
d_stepsize = 10
startpoint = 20
order = 10
#the dimension of the array needed is given Complete homogeneous symmetric 
variables = np.zeros(2)
numbofterms = sp.special.comb(len(variables) + order,order,exact=True)    
beta = np.zeros((int(datapoints/d_stepsize),numbofterms))

#Error arrays
temp = int((datapoints-startpoint)/d_stepsize)+1
MSE_test = np.zeros(temp)
BIAS = np.zeros(temp)
var = np.zeros(temp)

for i in range(startpoint,datapoints+d_stepsize,d_stepsize):
    curren_listel = int((i-startpoint)/(d_stepsize))
    print(curren_listel)
    # Make data.
    x = np.arange(0, 1, 1/i)
    y = np.arange(0, 1, 1/i)
    #scaling
    if scaling == True: 
        x -= np.mean(x)
        y -= np.mean(y)
    x, y = np.meshgrid(x,y)

    z = np.concatenate(FrankeFunction(x, y),axis=None)
    if noise == True: z += sigma*np.random.randn(len(z))

    #bring variables in the right form
    variables=[x,y]
    ##Designe Matrix
    A = DM(variables,order)
    #split data into test & train
    A_train, A_test, z_train, z_test = train_test_split(A, z, test_size=0.2)
    f_approx_test = np.empty((len(z_test), n_bootstrap+1))
    beta[curren_listel] = np.linalg.inv(A_train.T @ A_train) @ A_train.T @ z_train
    f_approx_test[:,0] = A_test @ beta[curren_listel][beta[curren_listel] != 0]
    for j in range(n_bootstrap+1):
        A_res, z_res = resample(A_train, z_train)
        beta[curren_listel] = np.linalg.inv(A_res.T @ A_res) @ A_res.T @ z_res
        f_approx_test[:,j] = A_test @ beta[curren_listel][beta[curren_listel] != 0]
    MSE_test[curren_listel] = np.mean( np.mean((z_test.reshape(-1, 1) - f_approx_test)**2, axis=1, keepdims=True) )
    #Calc BIAS & variance
    BIAS[curren_listel] = np.mean( (z_test.reshape(-1, 1) - np.mean(f_approx_test, axis=1, keepdims=True))**2 )
    var[curren_listel] = np.mean( np.var(f_approx_test, axis=1, keepdims=True) )

temp = np.arange(startpoint,datapoints+d_stepsize,d_stepsize)
plt.scatter(temp, BIAS, label='BIAS', color='orange', s=15)
plt.scatter(temp, MSE_test, label='test', color='blue', s=15)
plt.scatter(temp, var, label='var', color='red', s=15)     
plt.legend()
plt.show()



