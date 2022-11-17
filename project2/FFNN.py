#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import numpy as np
from functions import activation_functions as act_func
from functions import costfunctions

class FFNN():
    """ Class for our own Feed-Forward-Neural-Net code """
    def __init__(self, X_train = None, trainval = None,
                 h_layors = 1, h_neurons = 1, categories = None,
                 CostFunc = None, h_actf = act_func.sigmoid, o_actf = act_func.identity,
                 methode = "static", learningrate = 0.01):
        """ 
        Constructor for generating an instance of the class.
        
        Arguments
        
        ---------
        X_train: array
            train data values (default: None)
        trainval: array
            train function values (default: None)
        h_layors: int
            Number of hidden layors (atleast one) in the NN (default: 1)
        h_neurons: int
            Number of neurons in the hidden layors (default: None)
        categories: int
            Number of output categores (default: None)
        CostFunc: class
            Class containing the cost function and its derivative used for the backpropagation
        h_actf: class
            Class containing the activation function and its derivative for the hidden layors (default: sigmoid)
        o_actf: class
            Class containing the activation function and its derivative for the output layor (default: identity)
        methode: string
            Methode used for the stochastic gradient decent when updating bias and weights (default: static)
        learningrate: float
            Learningrate used for the stochastic gradient decent when updating bias and weights  (default: 0.01)
        
        Errors
        ------
        
        """
    
        self.h_neurons = h_neurons
        self.h_layors = h_layors
        self.X_train = X_train
        self.trainval = trainval
        self.categories = categories
        
        self.h_actf = h_actf
        self.o_actf = o_actf
        self.costfunc = CostFunc

        self.methode = methode
        self.learningrate = learningrate
        
        #Construct the initial bias and weight values
        self.weights, self.bias = self.bias_and_weights()
        
        
    def bias_and_weights(self):
        """Construct the starting values for bias and weights """
        
        h_neurons = self.h_neurons
        
        weights = []
        bias = []
        
        ###BIAS and WEIGHTS for the hidden layors###
        
        #first hidden layor
        np.random.seed(1999) #ensures reproducibility
        temp1 = np.random.randn(self.X_train.shape[1], h_neurons)
        temp2 = np.random.randn(h_neurons) + 0.01
        weights.append(temp1)
        bias.append(temp2)
        
        #all other hidden layors
        for _ in range(1,self.h_layors):
            temp1 = np.random.randn(h_neurons, h_neurons)
            temp2 = np.random.randn(h_neurons) + 0.01
            weights.append(temp1)
            bias.append(temp2)
        
        ###BIAS and WEIGHTS for the output layor###
        
        np.random.seed(1999) #ensures 
        temp1 = np.random.randn(h_neurons, self.categories)
        temp2 = np.zeros(self.categories) + 0.1
        weights.append(temp1)
        bias.append(temp2)
        return weights,bias
        
    def FF(self, X=None):
        
        h_layors = self.h_layors
        if X is None: X=self.X_train
        
        a, z = [],[]
        
        #Input Layor
        z.append(X)
        a.append(X)
        
        #hidden_layor
        for i in range(0,h_layors):
            z.append( a[i]@self.weights[i]+self.bias[i] )
            h_actf = self.h_actf(z[i+1])
            a.append( h_actf.func() )
            
        #output_layor
        z.append(a[h_layors]@self.weights[h_layors]+self.bias[h_layors])
        o_actf = self.o_actf(z[h_layors+1])
        a.append(o_actf.func())
        
        return z,a
    
    def backpropagation(self, z, a):
        
        delta, weight_grad, bias_grad = [],[],[]
        
        weights = self.weights
        h_layors = self.h_layors
        
        #Output Layor
        costfunc = self.costfunc(self.trainval,a[h_layors+1])
        gradient = costfunc.derivative()
        
        o_actf = self.o_actf(z[h_layors+1])
        funcgrad = o_actf.derivative()
        
        delta.append(gradient*funcgrad)
        
        for l in range(0,h_layors):
            h_actf = self.h_actf(z[h_layors-l])
            funcgrad = h_actf.derivative()
            
            temp = delta[l] @ weights[h_layors - l].T
            
            delta.append(temp * funcgrad)
        
        for i in range(len(self.weights)):
            weight_grad.append(a[h_layors-i].T @ delta[i])
            bias_grad.append(np.mean(delta[i],axis = 0))
        
        self.weight_grad = weight_grad.copy()
        self.bias_grad = bias_grad.copy()
        
        return None
       
    def update_WandB(self, w_upd = None, b_upd = None, delta_momentum = None):
        match self.methode:
            
            case "const": 
                for i in range(self.h_layors+1):
                    self.weights[i] -= self.learningrate*self.weight_grad[self.h_layors-i]
                    self.bias[i] -= self.learningrate * self.bias_grad[self.h_layors-i]
                return None
            
            case "momentum":
                w_newupd = []
                b_newupd = []
                for i in range(self.h_layors+1):
                    match w_upd:
                        case None:
                            temp1 = self.learningrate*self.weight_grad[self.h_layors-i]
                            temp2 = self.learningrate*self.bias_grad[self.h_layors-i]
                        case _:
                            temp1 = self.learningrate*self.weight_grad[self.h_layors-i]+delta_momentum*w_upd[i]
                            temp2 = self.learningrate*self.bias_grad[self.h_layors-i]+delta_momentum*b_upd[i]
                    w_newupd.append(temp1)
                    b_newupd.append(temp2)
                    self.weights[i] -= w_newupd[i]
                    self.bias[i] -= b_newupd[i]
                return w_newupd, b_newupd
    
from gen_data import functions
dimension = 1
coef = [3.5,3.0,4.0]
order = 2
batches = 1

func = functions(order = order, dimension=dimension, sigma=0.0,
                 coef=coef,points= 6)
data, funcval = func.polynomial()

costfunc = costfunctions.CostOLS
#data = np.array_split(data,batches,axis=0)
#data = np.array_split(trainval,batches)

nn = FFNN(X_train = data, trainval = funcval,
             h_layors = 1, h_neurons = 20, categories = 1,
             CostFunc = costfunc, 
             h_actf = act_func.identity,
             o_actf = act_func.identity,
             methode = "momentum", learningrate = 0.1)

epochs = 200
w_upd = None
b_upd = None
for itera in range(epochs):
    for i in range(batches):
        z,a = nn.FF()
        nn.backpropagation(z,a)
        w_upd, b_upd = nn.update_WandB(w_upd,b_upd,delta_momentum=0.8)
z,a = nn.FF()
print(funcval)
print(a[1+1])
print(np.mean(np.power((funcval-a[2]),2)))
