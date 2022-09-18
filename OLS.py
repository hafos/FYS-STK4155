from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from frankiefunction import FrankeFunction
import pandas as pd

# Perform a standard OLS regression analysis 

# Make grid / generate datapoints 
nx, ny = 100, 100 
x = np.arange(0, 1, nx)
y = np.arange(0, 1, nx)
x, y = np.meshgrid(x,y)

z = FrankeFunction(x, y, noise=False)

def DesignMatrix(x, y, p): 
    """ Defines the matrix X 
        If SOMETHING it should scale 

        Args:
            x 
            y
            p: Order of polynomial 

        Returns:
            X: Design matrix 

    """

    # TO DO : Add scale 

    # Flatten array using ravel ? 
    if len(x.shape) > 1: 
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    a = int((p+1)*(p+2)/2) # Number of elements in beta 
    X = np.ones((N,a))

    cols = [r'$x^0 y^0$']
    for i in range(1, p+1): 
        q = int(i + (i+1)/2)
        for k in range(i+1): 
            X[:,q+k] = x**(i-k) * y**k 
            cols.append(r'$x^{%i} y^{%i}$'%((i-k), k))

    # Nicer print 
    DesignMatrix = pd.DataFrame(X)
    print(cols) # Check that it looks good 
    # DesignMatrix.index = z
    # DesignMatrix.columns = ['1', 'z', 'z^(2/3)', 'z^(-1/3)', '1/z']

    return X 


X = DesignMatrix(x, y, p=5)

print(X)

def beta(X, y): 
    """ Uses matrix inversion to find beta (y = X*beta + epsilon)
    """
    # Train model 

    b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) # Evt dette? 
    # beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Energies)

    b = np.linalg.pinv(X.T @ X) @ X.T @ y

    return b


from sklearn.model_selection import train_test_split
X_train, X_test, z_train, z_test = train_test_split(X, z.ravel(), test_size=0.2)










# and then make the prediction
# ytilde = X @ beta # @ Matrix multiplixation XXX __matmul__ eller np.dot(X, beta) ? 

# Least squares in numpy 
# fit = np.linalg.lstsq(X, Energies, rcond =None)[0]
# ytildenp = np.dot(fit,X.T)





# Tin Bider, Algerie 