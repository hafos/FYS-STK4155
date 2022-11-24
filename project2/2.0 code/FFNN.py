#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
""" 

import numpy as np

class FFNN:

    def __init__(self, parameters, cost_func) -> None:
        self.cost_func = cost_func
        self.parameters = parameters
        self.layers = list()
        
    
    def add_layer(self, layer) -> None:
        if len(self.layers) > 0:
            parameters = self.layers[-1].size
        else:
            parameters = self.parameters

        # initializes weights and biases
        layer.create_weights(parameters)
        layer.create_bias()
        
        
        self.layers.append(layer)


    def feed_forward(self, inputs) -> (list, list):

        temp = inputs
        z = [inputs]
        a = [inputs]
        
        for layer in self.layers:

            temp, z = layer.forward(temp)
            
            z.append(z)
            a.append(temp)
        
        return z, a

    def back_prop(self, z, a, f_train, eta = 0.1) -> list:
            delta = list()
            temp = np.multiply(self.cost_func.grad(f_train, a[-1]), self.layers[-1].act_fn.grad(z[-1]))
            delta.append(temp)
            
            i = 0
            for layer in reversed(self.layers[:-1]):
                temp = ( delta[-1-i] @ layer.weights[-1-i].T ) * layer.act_fn.grad(z[-2-i])
                delta.append(temp)
                i -= 1
            
            return delta
        

    def train(self, X_train, f_train, eta= 0.1, epochs = 200, batches = 32, l2= 0) -> None:
        
        split_X_train = np.array_split(X_train,batches,axis=0)
        split_f_train = np.array_split(f_train,batches)
        
        np.random.seed(1999)
        for i in range(epochs):
            for _ in range(batches):
                rd_ind = np.random.randint(batches)
                z, a = self.feed_forward(split_X_train[rd_ind])
                delta = self.back_prop(z, a, split_f_train[rd_ind], eta = eta)
                
                layers[-1].weights -= eta * ( a[-1].T @ delta[0] + 2*l2*layers[-1].weights )
                layer[-1].bias -= eta * np.mean(delta[0],axis = 0)
                
                j = 1
                for layer in reversed(self.layers[:-1]):
                    layer.weights -= eta * ( a[-1-j].T @ delta[j] )
                    layer.bias -= eta * np.mean(delta[j],axis = 0)
                    j += 1
       
        return layer[-1].weights