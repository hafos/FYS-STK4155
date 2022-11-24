#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
""" 

import numpy as np

class layer():
    "Class to create a hidden or output layor for the FFNN"
    def __init__(self, neurons = None, act_fn = None, bias_const = 0.01, createbias = "random") -> None:
        """ 
        Constructor for generating an instance of the class.
        
        Arguments
        
        ---------
        neurons: int
            number of neurons / categories of the layor (default: None)
        act_fn: class
            class for the activation function of the layor containing 
            the function and the derivative (default: None)
        bias_const: float
            Constant that gets added to every bias (default: 0.01)
        creatbias: string
            Choose how the intial biases get created (default: "random")
        """
        
        self.neurons = neurons
        self.act_fn = act_fn
        self.weights = None
        self.bias_const = bias_const
        self.createbias = createbias
        self.bias = None


    def create_weights(self, parameters = None) -> None:
        """
        Initialises the weights for the layor
        
        Arguments
        
        ---------
        parameters: int
            Number of parameters of the previous layor (default: None)
        """  
        
        np.random.seed(1999)
        self.weights = np.random.randn(parameters, self.neurons)
        
    def create_bias(self) -> None:
        """
        Initialises the bias for the layor
        """
        
        match self.createbias:
            case "random":
                np.random.seed(1999)
                self.bias = np.random.randn(self.neurons) + self.bias_const
            case "zeros":
                self.bias = np.zeros(self.neurons) + self.bias_const
            case "ones":
                self.bias = np.ones(self.neurons) + self.bias_const
            case _:
                raise TypeError("valid values for <createbias> are: random, zeros, ones")
        

    def forward(self, inputs = None) -> (np.matrix, np.matrix):
        """
        Calcuates output output and activated output of the layor
        
        Arguments
        
        ---------
        inputs: np.matrix
            Output of the previous layor (default: None)
        """

        z = (inputs @ self.weights + self.bias)
        
        return z, self.act_fn.func(z)