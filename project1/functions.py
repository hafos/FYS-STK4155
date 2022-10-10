import numpy as np
from sklearn.preprocessing import PolynomialFeatures

## Designe Matrix
def DM(variables,order):
    poly = PolynomialFeatures(degree=order)
    A = poly.fit_transform(np.concatenate(np.stack([variables[i] for i in range(0,len(variables))],axis=-1),axis=0))
    return(A)

## R2 function
def R2(func, func_app):
    mean = np.mean(func)
    R_two=-np.sum(np.power(func-func_app,2))
    R_two/=np.sum(np.power(np.subtract(func,mean),2))
    R_two+=1  
    return R_two 

def get_MSE(y, y_tilde): 
    mse = 1/np.size(y) * np.sum((y - y_tilde)**2)
    n = np.size(y)
    # 1/len(z_test)*np.sum(np.power(z_test - ytilde_test, 2))
    return np.sum((y-y_tilde)**2)/n

def FrankeFunction(x,y,sigma):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    values = term1 + term2 + term3 + term4 
    values = np.concatenate(values,axis=None) + sigma*np.random.randn(len(x)*len(y))
    return values


