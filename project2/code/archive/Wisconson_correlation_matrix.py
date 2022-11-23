#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: simon hille
"""


from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

cancer = load_breast_cancer()

# Making a data frame
cancerpd = pd.DataFrame(cancer.data, columns=cancer.feature_names)



correlation_matrix = cancerpd.corr().round(1)
# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
plt.figure(figsize=(15,8))
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()