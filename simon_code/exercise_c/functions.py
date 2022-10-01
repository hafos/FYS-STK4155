import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#Designe Matrix
def DM(variables,order):
    poly = PolynomialFeatures(degree=order)
    A = poly.fit_transform(np.concatenate(np.stack([variables[i] for i in range(0,len(variables))],axis=-1),axis=0))
    return(A)

##R2 function
def R2(func, func_app):
    mean = np.mean(func)
    R_two=-np.sum(np.power(func-func_app,2))
    R_two/=np.sum(np.power(np.subtract(func,mean),2))
    R_two+=1  
    return R_two 
