#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
""" 

import numpy as np

class layer():

    def __init__(self, neurons, act_fn, bias_const = 0.01, createbias = "random") -> None:

        self.neurons = neurons
        self.act_fn = act_fn
        self.weights = None
        self.bias_const = bias_const
        self.createbias = createbias
        self.bias = None


    def create_weights(self, parameters) -> None:
        np.random.seed(1999)
        self.weights = np.random.randn(self.parameters, self.neurons)
        
    def create_bias(self,parameters) -> None:
        match self.create_bias:
            case "random":
                self.bias = np.random.randn(self.neurons) + self.bias_const
            case "zeros":
                self.bias = np.zeros(self.neurons) + self.bias_const
            case "ones":
                self.bias = np.ones(self.neurons) + self.bias_const
        

    def forward(self, inputs) -> (np.matrix, np.matrix):
        z = (self._weights @ inputs.T + self._biases).T
        return z, self.act_fn(z)