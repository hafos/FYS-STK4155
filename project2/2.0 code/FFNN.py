#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
""" 

import numpy as np


class FFNN:
    "Class to create a FFNN"
    def __init__(self, parameters = None, cost_fn = None) -> None:
        """ 
        Constructor for generating an instance of the class.
        
        Arguments
        
        ---------
        parameters: int
            number parameters of the input (default: None)
        cost_fn: class
            class for the cost function containing
            the function and the derivative (default: None)
        """
        
        self.cost_fn = cost_fn
        self.parameters = parameters
        self.layers = list()
        
    
    def add_layer(self, layer = None) -> None:
        """
        Adds a layor to NN
        
        Arguments
        ----------
        layer : class
            instance of the layer class (default: None)
        """

        if len(self.layers) > 0:
            parameters = self.layers[-1].neurons
        else:
            parameters = self.parameters

        #create weights and bias of the layor
        layer.create_weights(parameters)
        layer.create_bias()
        self.layers.append(layer)


    def feed_forward(self, inputs) -> (list, list):
        """
        For a given layor feed through the NN
        
        Arguments
        ----------
        input : np.matrix
            input that is feed through the NN (default: None)
        """
        
        temp_a = inputs
        z = [inputs]
        a = [inputs]
        
        for layer in self.layers:

            temp_z, temp_a = layer.forward(temp_a)
            
            z.append(temp_z)
            a.append(temp_a)
        
        return z, a

    def back_prop(self, z = None, a = None, f_train = None) -> list:
        """
        Calcuate the deltas needed for the SGD with back propagation
        
        Arguments
        ----------
        z : list
            list containing outputs from every layor (default: None)
        a : list
            list containing activated outputs from every layor (default: None)
        f_train : list
            list containing the values with which the NN is trained (default: None)
        """
            
        delta = list()
    
        temp = self.cost_fn.grad(f_train, a[-1]) * self.layers[-1].act_fn.grad(z[-1])
        delta.append(temp)
        
        i = 0
        for layer in reversed(self.layers[:-1]):
            temp = ( delta[i] @ self.layers[-1-i].weights.T ) * layer.act_fn.grad(z[-2-i])
            delta.append(temp)
            i += 1
        
        return delta
        

    def train(self, X_train = None, f_train = None, eta= 0.1, epochs = 200, batches = 32, l2= 0.0) -> None:
        """
        Train the NN with SGD with a constant learning rate
        
        Arguments
        ----------
        X_train : np.array
            Train data (default: None)
        f_train : list
            list containing the values with which the NN is trained (default: None)
        eta: float
            learningrate (default: 0.1)
        epochs: int
            number of epochs (default: 200)
        batches: int
            number of batches (default: 32)
        l2: float
            L2 regularization parameter (default: 0.0)
        """
        
        split_X_train = np.array_split(X_train,batches,axis=0)
        split_f_train = np.array_split(f_train,batches)
        
        np.random.seed(1999)
        for i in range(epochs):
            for _ in range(batches):
                rd_ind = np.random.randint(batches)
                z, a = self.feed_forward(split_X_train[rd_ind])
                delta = self.back_prop(z, a, split_f_train[rd_ind])
                
                j = 0
                for layer in reversed(self.layers):
                    layer.weights -= eta * ( a[-2-j].T @ delta[j] + 2*l2*layer.weights)
                    layer.bias -= eta * np.mean(delta[j],axis = 0)
                    j += 1
       
        return self.layers[-1].weights
    
    def reset(self):
        """
        Resets all layors in the NN
        """
        
        for layer in self.layers:
            layer.create_weights(layer.weights.shape[0])
            layer.create_bias()
            