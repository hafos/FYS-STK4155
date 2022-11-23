#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""

import numpy as np
from cost_act_func import activation_functions as act_func
from cost_act_func import CostOLS

class FFNN():
    """ Class for our own Feed-Forward-Neural-Net code """
    def __init__(self, X_train = None, trainval = None,
                 h_layors = 1, h_neurons = 1, categories = None, createbias = "random",
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
        createbias: string ("random or "ones" or "zeros")
            How the initial biases get created (default: random)
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
        self.createbias = createbias
        
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
        
        ###WEIGHTS###
        
        #first hidden layor
        np.random.seed(1999) #ensures reproducibility
        temp1 = np.random.randn(self.X_train.shape[1], h_neurons)
        weights.append(temp1)

        #all other hidden layors
        np.random.seed(1999) #ensures 
        for _ in range(1,self.h_layors):
            temp1 = np.random.randn(h_neurons, h_neurons)
            weights.append(temp1)
        
        #output layor
        np.random.seed(1999) #ensures 
        temp1 = np.random.randn(h_neurons, self.categories)
        weights.append(temp1)
        
        ###BIAS###
        
        match self.createbias:
            case "random":
                #first hidden layor
                np.random.seed(1999) #ensures reproducibility
                temp2 = np.random.randn(h_neurons) + 0.01
                bias.append(temp2)
                #all other hidden layors
                for _ in range(1,self.h_layors):
                    temp2 = np.random.randn(h_neurons) + 0.01
                    bias.append(temp2)
                #output layor
                temp2 = np.random.randn(self.categories) + 0.1
                bias.append(temp2)
                
            case "ones":
                #first hidden layor
                temp2 = np.ones(h_neurons) + 0.01
                bias.append(temp2)
                #all other hidden layors
                for _ in range(1,self.h_layors):
                    temp2 = np.ones(h_neurons) + 0.01
                    bias.append(temp2)
                #output layor
                temp2 = np.ones(self.categories) + 0.1
                bias.append(temp2) 
                
            case "zeros":
                #first hidden layor
                temp2 = np.zeros(h_neurons) + 0.01
                bias.append(temp2)
                #all other hidden layors
                for _ in range(1,self.h_layors):
                    temp2 = np.zeros(h_neurons) + 0.01
                    bias.append(temp2)
                #output layor
                temp2 = np.zeros(self.categories) + 0.1
                bias.append(temp2)
            
            case _:
                raise TypeError("input for <createbias> is invalid")
        
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
            h_actf = self.h_actf
            a.append( h_actf.func(z[i+1]) )
        
            
        #output_layor
        z.append(a[h_layors]@self.weights[h_layors]+self.bias[h_layors])
        o_actf = self.o_actf
        a.append(o_actf.func(z[h_layors+1]))
        
        return z,a
    
    def backpropagation(self, z, a, trainval, hyperpar = 0.0):
        
        delta, weight_grad, bias_grad = [],[],[]
        
        weights = self.weights
        h_layors = self.h_layors
        
        #Output Layor
        costfunc = self.costfunc
        gradient = costfunc.derivative(trainval,a[h_layors+1])
        
        o_actf = self.o_actf
        funcgrad = o_actf.derivative(z[h_layors+1])
        
        delta.append(gradient*funcgrad)
        
        for l in range(0,h_layors):
            h_actf = self.h_actf
            funcgrad = h_actf.derivative(z[h_layors-l])
            
            temp = delta[l] @ weights[h_layors - l].T
            
            delta.append(temp * funcgrad)
        
        for i in range(len(self.weights)):
            weight_grad.append(a[h_layors-i].T @ delta[i] + 2*hyperpar*weights[h_layors-i])
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
                return self.weights
            
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
    

