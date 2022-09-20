import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from frankefunction import FrankeFunction # XXX rename this correctly 
from functions import DesignMatrix, get_beta, MSE, R2

# Perform a standard OLS regression analysis in x and y up to fifth order 

# Make grid / generate datapoints 
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
print(np.shape(x))
x, y = np.meshgrid(x,y)
print(np.shape(x))

z = FrankeFunction(x, y, noise=False)
# Scaling by subtracting mean value 
# Q : Skal man skalere x of y eller z ?
# x = scale(x, axis = 1)
# y = scale(y, axis = 0)

def OLS(x, y, z, p=5): 
    X = DesignMatrix(x, y, p)

    # Split into training and testing 
    ratio = 0.2
    X_train, X_test, z_train, z_test = train_test_split(X, np.ravel(z), test_size=ratio)

    return X_train, X_test, z_train, z_test 

# Plot the resulting scores (MSE and R²) as functions of the polynomial degree 

def plot_OLS(x,y,z, p): 
    """ Plots the MSE and R² scores as functions of the polynomial degree 
    
    """
    X = DesignMatrix(x, y, p)
    X_train, X_test, z_train, z_test = OLS(x, y, z, p)
    beta = get_beta(X_train, z_train)
    z_tilde_train = X_train @ beta # predicted z 
    z_tilde_test  = X_test  @ beta # predicted z 

    # Create arrays 
    train_MSE = np.zeros(p)
    test_MSE  = np.zeros(p)
    train_R2  = np.zeros(p)
    test_R2   = np.zeros(p)
    p_array = np.zeros(p)
    i = 0

    for n in range(1, p+1):
        print(f"Order {n}")
        z_all = X @ get_beta(X, z.ravel())
        z_all = np.reshape(z_all, (np.shape(x)[0], np.shape(x)[0]))
        
        # fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        # Plot the surface.
        # surf = ax.plot_surface(x, y, z_all, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(-0.10, 1.40)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # fig.suptitle(f"Order {n}", fontsize=20)

        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        train_MSE[i] = MSE(z_train.ravel(), z_tilde_train.ravel())
        test_MSE[i]  = MSE(z_test.ravel(), z_tilde_test.ravel())
        p_array[i] = n
        i+=1


    fig, ax = plt.subplots()
    ax.plot(p_array, train_MSE, label="Train MSE")
    ax.plot(p_array, test_MSE, label="Test MSE")
    ax.set_yscale('log')
    ax.legend()
    plt.show()

p = 5
plot_OLS(x,y,z, p)



# Plot the parameters beta for increasing order of the polynomial 



# Least squares in numpy 
# fit = np.linalg.lstsq(X, z, rcond =None)[0]
# ytildenp = np.dot(fit,X.T)

# Tin Bider, Algerie 

