#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
""" 

import numpy as np

class rescale:
    def submean(X_train = None, X_test = None, y_train = None, y_test = None):        
        mean_X = np.mean(X_train, axis=0)
        std_X = np.std(X_train, axis=0)
        X_train_scaled = (X_train - mean_X) / std_X
        X_test_scaled = (X_test - mean_X) / std_X

        mean_y = np.mean(y_train)
        std_y = np.std(y_train)
        y_train_scaled = (y_train - mean_y) / std_y
        y_test_scaled = (y_test - mean_y) / std_y
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled