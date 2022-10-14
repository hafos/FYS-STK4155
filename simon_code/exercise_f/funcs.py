#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:34:20 2022

@author: simon
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn import linear_model



class functions:
    def __init__(self):
        self.param = 0
        self.rescale = False
        
    def makedata(points,sigma, func):
        #ensures that we get the same data in loops
        np.random.seed(1999)
        x = np.random.rand(points)
        y = np.random.rand(points)
        x, y = np.meshgrid(x,y)
        fval = np.concatenate(func(x,y,sigma),axis=None)
        return x,y,fval
    
    #Defines the FrankeFunction
    def FrankeFunction(x,y,sigma):
        np.random.seed(1999)
        #ensures that we get the same data in loops
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        values = term1 + term2 + term3 + term4 
        values += sigma*np.random.randn(len(x),len(y))
        return values
   
    #Designe Matrix
    def DM(variables,order):
        poly = PolynomialFeatures(degree=order)
        A = poly.fit_transform(np.concatenate(np.stack([variables[i] for i in range(0,len(variables))],axis=-1),axis=0))
        return(A)

    ##R2 function
    def R2(fval, faval):
        mean = np.mean(fval)
        R_two=-np.sum(np.power(fval-faval,2))
        R_two/=np.sum(np.power(np.subtract(fval,mean),2))
        R_two+=1  
        return R_two 
     
    ##define all models
    def OLS(A_train,f_train):
        beta = np.linalg.pinv(A_train.T @ A_train) @ A_train.T @ f_train
        return beta
    
    def Ridge(A_train,f_train):
        beta = np.linalg.pinv(A_train.T @ A_train + functions.param*np.identity(A_train.T.shape[0])) 
        beta = beta @ A_train.T @ f_train
        return beta
    
    def Lasso(A_train,f_train):
        RegLasso = linear_model.Lasso(functions.param, fit_intercept = False)
        RegLasso.fit(A_train,f_train)
        beta = beta = RegLasso.coef_
        return beta
        
    def bootstrap(n,A_train,A_test,f_train,f_test,model):
        #Need this for the case we do not rescale -> add zero to fapproox
        f_scale = 0
        score = np.zeros(n)
        fte_aval = np.zeros((len(f_test),n))
        for j in range(0,n):
            ATr_res, ftr_res = resample(A_train, f_train, random_state = j)
            ATe_res = A_test + 0
            ##rescaling
            if functions.rescale == True:
                ATe_res -= np.mean(ATr_res,axis=0)
                ATr_res -= np.mean(ATr_res,axis=0)
                f_scale = np.mean(ftr_res)         
                ftr_res -= f_scale   
            beta = model(ATr_res, ftr_res)
            fte_aval[:,j] = ATe_res @ beta + f_scale
            score[j] = np.mean((f_test-fte_aval[:,j])**2)
        return np.mean(score), fte_aval
    
    def crossval(folds,A,fval,model):
        #Need this for the case we do not rescale -> add zero to fapproox
        f_scale = 0
        score = np.zeros(folds)
        rs = KFold(n_splits=folds,shuffle=True, random_state=1)
        j = 0
        for train_index, test_index in rs.split(A):
            A_train = A[train_index] + 0
            f_train = fval[train_index] + 0
            A_test = A[test_index] + 0
            f_test = fval[test_index]
            if functions.rescale == True:
                A_test  -= np.mean(A_train,axis=0)
                A_train -= np.mean(A_train,axis=0)
                f_scale = np.mean(f_train)
                f_train -= f_scale
            beta = model(A_train, f_train)
            fte_aval = A_test @ beta + f_scale
            score[j] = np.sum((fte_aval - f_test)**2)/np.size(fte_aval)
            j +=1
        return np.mean(score)
    
    
    
    
    
    
    