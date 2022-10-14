import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

from funcs import functions as fnc
from sklearn.model_selection import train_test_split

np.random.seed(2018)



def OLS(points, sigma, max_order, bootstrap=False): 
    # Initialize data and design matrix 
    x, y, fval = fnc.makedata(points, sigma, fnc.FrankeFunction)
    variables = [x,y]
    A = fnc.DM(variables, max_order)
    fnc.rescale = False
    fscale = 0 # Rescaling of the data 

    # Split data into test & train
    A_train, A_test, f_train, f_test = train_test_split(A, fval, test_size=0.2, random_state = 1)

    # Initialize arrays
    MSE_test  = np.zeros(max_order)
    MSE_train = np.zeros(max_order)
    R2_test   = np.zeros(max_order)
    R2_train  = np.zeros(max_order)
    BIAS = np.zeros(max_order)
    var = np.zeros(max_order)
    ftr_aval = np.zeros(len(f_train)) # y tilde???? 
    fte_aval = np.zeros(len(f_test)) 

    # Dimension of array is given complete homogeneous symmetric polynomial
    numbofterms = sp.special.comb(len(variables) + max_order, max_order, exact=True) 
    beta = np.full((max_order, numbofterms), np.nan)

    for i in range(1, max_order+1):
        #current number of tearms also via complete homogeneous symmetric polynomials
        currentnot = sp.special.comb(len(variables) + i, i, exact=True)
        #select only the terms of the full desinge matrix needed for the current order
        ATrCur = A_train[:,0:currentnot] + 0 
        ATeCur =  A_test[:,0:currentnot] + 0

        if bootstrap==True: 
            #for fig 2.11 of Hastie, Tibshirani, and Friedman
            if fnc.rescale == True:
                ATrCur -= np.mean(ATrCur)
                fscale = np.mean(f_train)
            beta = fnc.OLS(ATrCur,f_train-fscale)
            ftr_aval = ATrCur @ beta + fscale

            #Calcuate the errors
            MSE_test[i-1], MSE_test[i-1], ftr_aval, fte_aval = fnc.bootstrap(n_bootstrap, ATrCur, ATeCur, f_train, f_test, fnc.OLS)
            #MSE_train[i-1] = np.mean(np.power(f_train-ftr_aval,2))
            BIAS[i-1] = np.mean( (f_test.reshape(-1, 1) - np.mean(fte_aval, axis=1, keepdims=True))**2 )
            var[i-1] = np.mean( np.var(fte_aval, axis=1, keepdims=True) )
            print(f"MSE - BIAS + var {(MSE_test[i-1] - (BIAS[i-1] + var[i-1]))}")
        
        else: 
            #calc both errors and store the betas in the process
            beta[i-1][:currentnot] = fnc.OLS(ATrCur,f_train)
            fte_aval = ATeCur @ beta[i-1][~np.isnan(beta[i-1])]
            ftr_aval = ATrCur @ beta[i-1][~np.isnan(beta[i-1])]
            MSE_test[i-1] = np.mean(np.power(f_test-fte_aval,2))
            MSE_train[i-1] = np.mean(np.power(f_train-ftr_aval,2))
            R2_test[i-1] = fnc.R2(f_test, fte_aval)
            R2_train[i-1] = fnc.R2(f_train, ftr_aval)

    return MSE_train, MSE_test, R2_train, R2_test, BIAS, var, beta





sigma     = 0.1 # No noise for sigma = 0 
points    = 40  # Number of x & y points, total amount of datapoints is this squared
max_order = 5   # Program will fit a polnyomial up to this order

MSE_train, MSE_test, R2_train, R2_test, BIAS, var, beta = OLS(points, sigma, max_order)

## Plots 
p_arr = np.arange(1, max_order+1, 1.0)

fig, ax = plt.subplots(nrows=2, sharex=True)
# Data 
ax[0].plot(p_arr, MSE_train, label="Train MSE", color="orange")
ax[1].plot(p_arr, R2_train, label="Train R2", color="blue")
# Predicted 
ax[0].plot(p_arr, MSE_test, label="Test MSE", color="orange") 
ax[1].plot(p_arr, R2_test, label="Test R2", color="blue")
ax[0].legend()
ax[1].legend()
fig.tight_layout()
plt.show()

for i in range(0,len(beta[0])):
    plt.scatter(p_arr, beta[0:,i], s=15)
plt.show()


sigma = 0.1
points = 30
max_order = 18
n_bootstrap = 50
fnc.rescale = True
MSE_train, MSE_test, R2_train, R2_test, BIAS, var, beta = OLS(points, sigma, max_order, bootstrap=True)

##reproduce fig 2.11 of Hastie, Tibshirani, and Friedman
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_train, label='train', color='orange', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_test, label='test', color='blue', s=15)
plt.legend(loc = "upper center")
plt.show()

sigma = 0.1
points = 30
max_order = 18
n_bootstrap = 50
fnc.rescale = False
MSE_train, MSE_test, R2_train, R2_test, BIAS, var, beta = OLS(points, sigma, max_order, bootstrap=True)
##Bias Variance tradeoff
plt.scatter(np.arange(1, max_order+1, 1.0), BIAS, label='BIAS', color='orange', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), MSE_test, label='test', color='blue', s=15)
plt.scatter(np.arange(1, max_order+1, 1.0), var, label='var', color='red', s=15)     
plt.legend(loc = "upper center")
plt.show()