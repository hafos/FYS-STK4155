from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from frankiefunction import FrankeFunction

# Perform a standard OLS regression analysis 


# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)


def X(x, y, n=5): 
    """ Defines the matrix X up to order n = 5 

    """

    # Create a column of ones 
    Ones = np.ones((len(x),))

    ol = np.ones_like(z)
    # print(ol)

    # X = np.zeros(len(x))
    # ol = np.ones(len(x))

    if n == 1: 
        X = np.c_[ol,x,y]
    if n == 2: 
        X = np.c_[ol, x, y, x**2, y**2]
    if n == 3: 
        X = np.c_[ol, x, y, x**2, y**2, x**3, y**3]
    if n == 4: 
        X = np.c_[ol, x, y, x**2, y**2, x**3, y**3, x**4, y**4]
    if n == 5: 
        X = np.c_[ol, x, y, x**2, y**2, x**3, y**3, x**4, y**4, x**5, y**5]
    return X 


print(X(x, y, n=1))





# matrix inversion to find beta
# beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Energies)
# and then make the prediction
# ytilde = X @ beta # @ Matrix multiplixation XXX __matmul__ eller np.dot(X, beta) ? 

# Least squares in numpy 
# fit = np.linalg.lstsq(X, Energies, rcond =None)[0]
# ytildenp = np.dot(fit,X.T)



# y = X beta + epsilon 
