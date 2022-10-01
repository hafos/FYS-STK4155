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
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

#import all needed functions located in seperate documents
from FrankeFunction import FrankeFunction
from functions import *


np.random.seed(2018)

noise = True
n_bootstrap = 100
##kfold
k=10

# Make data.
x = np.arange(0, 1, 0.025)
y = np.arange(0, 1, 0.025)
sigma = 0.25
x, y = np.meshgrid(x,y)
z = np.concatenate(FrankeFunction(x, y),axis=None)
if noise == True: z += sigma*np.random.randn(len(z))

##Approximation
max_order = 15
#bring variables in the right form
variables=[x,y]
#the dimension of the array needed is given Complete homogeneous symmetric 
numbofterms = sp.special.comb(len(variables) + max_order,max_order,exact=True)

#Error arrays
MSE_bs = np.zeros(max_order)
scores_KFold = np.zeros(k)
MSE_KFOLD = np.zeros(max_order)
BIAS = np.zeros(max_order)
var = np.zeros(max_order)


##Full designe Matrix
A = DM(variables,max_order)
#split data into test & train
rs = KFold(n_splits=k,shuffle=True)
for i in range(1,max_order+1):
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)    
    j=0
    A_curr = A[:,0:currentnot]
    for train_index, test_index in rs.split(A_curr):
        A_train = A_curr[train_index]
        z_train = z[train_index]
        A_test = A_curr[test_index]
        z_test = z[test_index]
        beta = np.linalg.pinv(A_train.T @ A_train) @ A_train.T @ z_train
        f_approx_test= A_test @ beta
        scores_KFold[j] = np.sum((f_approx_test - z_test)**2)/np.size(f_approx_test)
        j +=1
    MSE_KFOLD[i-1] = np.mean(scores_KFold)
    A_train, A_test, z_train, z_test = train_test_split(A_curr, z, test_size=0.2)
    beta = np.linalg.inv(A_train.T @ A_train) @ A_train.T @ z_train
    f_approx_test = np.empty((len(z_test), n_bootstrap+1))
    f_approx_test[:,0] = A_test @ beta
    for j in range(n_bootstrap+1):
        A_res, z_res = resample(A_train, z_train)
        beta = np.linalg.inv(A_train.T @ A_train) @ A_train.T @ z_train
        f_approx_test[:,j] = A_test @ beta
    MSE_bs[i-1] = np.mean( np.mean((z_test.reshape(-1, 1) - f_approx_test)**2, axis=1, keepdims=True) )
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_bs, label='MSE_bs', color='orange', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_KFOLD, label='MSE_KFOLD', color='blue', s=15)    
plt.legend()
plt.show()
