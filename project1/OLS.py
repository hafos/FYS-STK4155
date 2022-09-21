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
# z /= np.max(z)


# Scaling by subtracting mean value 
# Q : Skal man skalere x of y eller z ?
x = scale(x, axis = 1)
y = scale(y, axis = 0)

def OLS(x, y, z, p=5): 
    X = DesignMatrix(x, y, p)

    # print(X)

    # Split into training and testing 
    ratio = 0.2
    X_train, X_test, z_train, z_test = train_test_split(X, np.ravel(z), test_size=ratio)

    # Scale the data ?? 
    X_train = X_train - np.mean(X_train)
    X_test = X_test - np.mean(X_test)

    beta = get_beta(X_train, z_train)
    z_tilde_train = X_train @ beta # fitted z 
    z_tilde_test  = X_test  @ beta # predicted z 

    # MSE 
    train_MSE = MSE(z_train.ravel(), z_tilde_train.ravel())
    test_MSE  = MSE(z_test.ravel(), z_tilde_test.ravel())
    # R2
    train_R2  = R2(z_train.ravel(), z_tilde_train.ravel())
    test_R2   = R2(z_test.ravel(), z_tilde_test.ravel())

    return X_train, X_test, z_train, z_test, train_MSE, test_MSE, train_R2, test_R2, beta

OLS(x,y,z,p=5)

# Plot the resulting scores (MSE and R²) as functions of the polynomial degree 

def plot_OLS(x,y,z, p): 
    """ Plots the MSE and R² scores as functions of the polynomial degree 
    
    """
    # X = DesignMatrix(x, y, p)
    

    # Create arrays 
    train_MSE_arr = np.zeros(p)
    test_MSE_arr  = np.zeros(p)
    train_R2_arr  = np.zeros(p)
    test_R2_arr   = np.zeros(p)
    beta_list = []
    p_array = np.linspace(1,p,p)
    i = 0

    for n in range(1, p+1):
        X_train, X_test, z_train, z_test, train_MSE, test_MSE, train_R2, test_R2, beta = OLS(x, y, z, p=n)
        print(f"Order {n}")
        # z_all = X @ get_beta(X, z.ravel())
        # z_all = np.reshape(z_all, (np.shape(x)[0], np.shape(x)[0]))
        
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
        # train_MSE[i] = MSE(z_train.ravel(), z_tilde_train.ravel())
        beta_list.append(beta)
        train_MSE_arr[i] = train_MSE
        test_MSE_arr[i]  = test_MSE
        train_R2_arr[i] = train_R2
        test_R2_arr[i]  = test_R2
        # p_array[i] = n
        i += 1

    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(p_array, train_MSE_arr, label="Train MSE")
    ax[0].plot(p_array, test_MSE_arr, label="Test MSE")

    ax[1].plot(p_array, train_R2_arr, label="Train R2")
    ax[1].plot(p_array, test_R2_arr, label="Test R2")
    # ax.plot(p_array, beta, label="beta")
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    plt.show()


p = 5
plot_OLS(x,y,z, p)



# Plot the parameters beta for increasing order of the polynomial 



# Least squares in numpy 
# fit = np.linalg.lstsq(X, z, rcond =None)[0]
# ytildenp = np.dot(fit,X.T)

# Tin Bider, Algerie 

