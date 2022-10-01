import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed 

#import all needed functions located in seperate documents
from FrankeFunction import FrankeFunction
from functions import *

noise = True

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)

x, y = np.meshgrid(x,y)

z = np.concatenate(FrankeFunction(x, y),axis=None)
if noise == True: z += 0.1*np.random.randn(len(z))

##Approximation
max_order = 5
#bring variables in the right form
variables=[x,y]
#the dimension of the array needed is given Complete homogeneous symmetric polynomial
numbofterms = sp.special.comb(len(variables) + max_order,max_order,exact=True)
beta = np.zeros((max_order,numbofterms))

#Error arrays
MSE = np.zeros(max_order)
R_two = np.zeros(max_order)

##Full designe Matrix
A = DM(variables,max_order)
#split data into test & train
A_train, A_test, z_train, z_test = train_test_split(A, z, test_size=0.2)

for i in range(1,max_order+1):
    currentnot = sp.special.comb(len(variables) + i,i,exact=True)
    print(currentnot)
    beta[i-1][:currentnot] = np.linalg.inv(A_train[:,0:currentnot].T @ A_train[:,0:currentnot]) @ A_train[:,0:currentnot].T @ z_train
    #ytilde
    f_approx = A_test[:,0:currentnot] @ beta[i-1][beta[i-1] != 0] #with calc beta
    #calcuate the two errors
    MSE[i-1] = 1/len(z_test)*np.sum(np.power(z_test-f_approx,2))
    R_two[i-1] = R2(z_test,f_approx)

plt.scatter(np.arange(1, max_order+1, 1.0), MSE, label='Data', color='orange', s=15)
plt.legend()
plt.show()

plt.scatter(np.arange(1, max_order+1, 1.0), R_two, label='Data', color='blue', s=15)
plt.legend()
plt.show()

for i in range(0,len(beta[0])):
    plt.scatter(np.arange(1,max_order+1,1.0), beta[0:,i], s=15)
plt.show()


