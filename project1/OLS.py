from xml.dom.minidom import Identified
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import sklearn.linear_model as skl 
from sklearn.preprocessing import StandardScaler

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

def OLS(X, z, p=5, scale=False): 
    # Split into training and testing 
    ratio = 0.2
    X_train, X_test, z_train, z_test = train_test_split(X, np.ravel(z), test_size=ratio)

    if scale == True: 
        clf = skl.LinearRegression().fit(X_train, z_train)

        print("MSE before scaling: {:.2f}".format(MSE(clf.predict(X_test), z_test)))
        print("R2 score before scaling: {:.2f}".format(R2(clf.predict(X_test), z_test)))

        # Scale the data 
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # print("Feature min values before scaling:\n {}".format(X_train.min(axis=0)))
        # print("Feature max values before scaling:\n {}".format(X_train.max(axis=0)))
        # print("Feature min values after scaling:\n {}".format(X_train_scaled.min(axis=0)))
        # print("Feature max values after scaling:\n {}".format(X_train_scaled.max(axis=0)))

        clf = skl.LinearRegression().fit(X_train_scaled, z_train)

        print("MSE after scaling: {:.2f}".format(MSE(clf.predict(X_test_scaled), z_test)))
        print("R2 score after scaling: {:.2f}".format(R2(clf.predict(X_test_scaled), z_test)))    

        X_train = X_train_scaled
        X_test = X_test_scaled

    beta = get_beta(X_train, z_train)
    z_tilde_train = X_train @ beta # fitted z 
    z_tilde_test  = X_test  @ beta # predicted z 

    return X_train, X_test, z_train, z_test, z_tilde_train, z_tilde_test, beta


def plot_OLS(x,y,z, p, scale=False): 
    """ Plots the MSE and R² scores as functions of the polynomial degree, and 
        the parameter beta for increasing order of the polynomial 
    
    """ 

    # Create arrays 
    train_MSE_arr = np.zeros(p)
    test_MSE_arr  = np.zeros(p)
    train_R2_arr  = np.zeros(p)
    test_R2_arr   = np.zeros(p)
    beta_list = []
    p_array = np.linspace(1,p,p)
    i = 0

    for n in range(1, p+1):
        print(f"\nOrder {n}")
        X = DesignMatrix(x, y, n)
        X_train, X_test, z_train, z_test, z_tilde_train, z_tilde_test, beta = OLS(X, z, n, scale)

        # MSE 
        train_MSE = MSE(z_train.ravel(), z_tilde_train.ravel())
        test_MSE  = MSE(z_test.ravel(), z_tilde_test.ravel())
        # R2
        train_R2  = R2(z_train.ravel(), z_tilde_train.ravel())
        test_R2   = R2(z_test.ravel(), z_tilde_test.ravel())
        
        z_all = X @ get_beta(X, z.ravel())
        z_all = np.reshape(z_all, (np.shape(x)[0], np.shape(x)[0]))
        
        # fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
        # # Plot the surface.
        # surf = ax.plot_surface(x, y, z_all, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # # Customize the z axis.
        # ax.set_zlim(-0.10, 1.40)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # fig.suptitle(f"Order {n}", fontsize=20)

        # # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        
        beta_list.append(beta)
        train_MSE_arr[i] = train_MSE
        test_MSE_arr[i]  = test_MSE
        train_R2_arr[i] = train_R2
        test_R2_arr[i]  = test_R2
        # p_array[i] = n
        i += 1
    
    # Plot the resulting scores (MSE and R²) as functions of the polynomial degree 
    # XXX Needs improvement, see Fig 2.11 
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(p_array, train_MSE_arr, label="Train MSE")
    ax[0].plot(p_array, test_MSE_arr, label="Test MSE")

    ax[1].plot(p_array, train_R2_arr, label="Train R2")
    ax[1].plot(p_array, test_R2_arr, label="Test R2")
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()

    # Plot the parameters beta for increasing order of the polynomial 
    # plt.plot(p_array, beta, label="beta")
    plt.show()


p = 5
plot_OLS(x,y,z, p, scale=True)



# Include resampling techniques 



def test_reg_anal(method, n): 
    """ Tests the regression analysis by asserting """
    # I = np.identity(n)
    I = DesignMatrix(x, y, n)
    I = np.identity(400)
    if method == "OLS": 
        X_train, X_test, z_train, z_test, z_tilde_train, z_tilde_test, beta  = OLS(I, z, n, scale=False)
        assert MSE(z_train, z_tilde_train) == 0 
        print("Testing the regression analysis", MSE(z_test, z_tilde_test)) # This does not hold, is it supposed to? 
    elif method == "Ridge": 
        pass 
    elif method == "Lasso": 
        pass 
    else: 
        pass 
test_reg_anal("OLS", 5)



# Tin Bider, Algerie 