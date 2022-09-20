import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

from frankefunction import FrankeFunction 
from functions import DesignMatrix, beta, MSE, R2

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


def plot_OLS(x,y,z, p): 
    """ Plots the MSE and R² scores as functions of the polynomial degree 
    
    """
    X_train, X_test, z_train, z_test = OLS(x, y, z, p)

    p_array = np.linspace(1, p, p)
    train_MSE = np.zeros(p)

    b = beta(X_train, z_train)
    z_ = X_train @ b # predicted z 

    
    beta_train = beta(X_train, z_train)
    beta_test  = beta(X_test,  z_test)
    # Predicted z 
    z_train_tilde = X_train @ b
    z_test_tilde  = X_test  @ b 

    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    # Plot the surface 
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    # ax.set_zlim(-0.05, 1.05)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.suptitle(f"Order {p}", fontsize=20)

    fig.colorbar(surf, shrink=0.5, aspect=5) # Add a color bar which maps values to colors.

    poly_deg = np.zeros(p)
    i = 0
    train_MSE[i] = MSE(z_train.ravel(), z_.ravel())
    poly_deg[i] = i
    i += 1

    ax.plot(poly_deg, train_MSE, label="Train MSE")
    ax.set_yscale('log')
    ax.legend()
    plt.show()

p = 5
for i in range(1,p+1): ''
# X = DesignMatrix(x, y, p)
plot_OLS(x,y,z, p=2)


# Plot the parameters beta for increasing order of the polynomial 



# Least squares in numpy 
# fit = np.linalg.lstsq(X, z, rcond =None)[0]
# ytildenp = np.dot(fit,X.T)

# Tin Bider, Algerie 