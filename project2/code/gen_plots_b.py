"""
Date: 23.11.22 

@author: semyaat
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import numpy as np 
import time

import sys
sys.path.append('src')

from FFNN import FFNN
from gen_data import functions
from cost_act_func import activation_functions as act_func
from cost_act_func import CostOLS

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.grid': True})
plt.rc('legend', frameon=False)
params = {'legend.fontsize': 25,
			'figure.figsize': (12, 9),
			'axes.labelsize': 25,
			'axes.titlesize': 25,
			'xtick.labelsize': 'x-large',
			'ytick.labelsize': 'x-large'}
pylab.rcParams.update(params)


def compare_nofl(): 
    """ Compares number of hidden layers """
    dimension = 2
    batches = 1

    func = functions(dimension=dimension, sigma=0.0,points= 100)
    data, funcval = func.FrankeFunction()

    costfunc = CostOLS


    neurons = np.arange(2,30)
    MSE = np.zeros((2,len(neurons)))

    i = 0
    for nr in neurons:
        nn1 = FFNN(X_train = data, trainval = funcval,
                h_layors = 1, h_neurons = nr, categories = 1,
                CostFunc = costfunc, 
                h_actf = act_func.sigmoid,
                o_actf = act_func.identity,
                methode = "const", learningrate = 0.1)
        nn2 = FFNN(X_train = data, trainval = funcval,
                h_layors = 2, h_neurons = nr, categories = 1,
                CostFunc = costfunc, 
                h_actf = act_func.sigmoid,
                o_actf = act_func.identity,
                methode = "const", learningrate = 0.1)
        epochs = 100
        for itera in range(epochs):
            for _ in range(batches):
                z1,a1 = nn1.FF()
                z2,a2 = nn2.FF()
                nn1.backpropagation(z1,a1)
                nn1.update_WandB()
                nn2.backpropagation(z2,a2)
                nn2.update_WandB()
        z1,a1 = nn1.FF()
        z2,a2 = nn2.FF()
        MSE[0,i] = CostOLS.func(funcval,a1[len(a1)-1])
        MSE[1,i] = CostOLS.func(funcval,a2[len(a2)-1])
        i += 1

    plt.scatter(neurons, MSE[0,:])
    plt.scatter(neurons, MSE[1,:])


def NN_Relu(): 
    dimension = 2
    epochs = 100
    batches = 300
    learningrate = 0.1

    func = functions(dimension=dimension, sigma=0.25, points= 100)
    data, funcval = func.FrankeFunction()

    learningrates = [1e-2, 1e-3, 1e-4]
    MSE = np.zeros((len(learningrates),1))

    split_data = np.array_split(data,batches,axis=0)
    split_funcval = np.array_split(funcval,batches)

    i = 0
    for lr in learningrates:
        costfunc = CostOLS
        nn = FFNN(X_train = data, trainval = funcval,
            h_layors = 1, h_neurons = 30, categories = 1,
            CostFunc = costfunc, 
            h_actf = act_func.relu,
            o_actf = act_func.identity,
            methode = "const", learningrate = lr)
            
        np.random.seed(1999) #ensures reproducibility
        for itera in range(epochs):
            for _ in range(batches):
                rd_ind = np.random.randint(batches)
                z,a = nn.FF(split_data[rd_ind])
                print(a[len(a)-1])   
                nn.backpropagation(z,a,split_funcval[rd_ind])
                nn.update_WandB()
        z,a = nn.FF()
        MSE[i] = costfunc.func(funcval,a[len(a)-1])
        i += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='1.3e')
    ax.set_xlabel("lambda")
    ax.set_ylabel("log$_{10}$(eta)")
    ax.set_xticklabels([1])
    ax.set_yticklabels(np.log10(learningrates))


def NN_sigmoid(): 

    dimension = 2
    epochs = 1000
    batches = 64

    func = functions(dimension=dimension, sigma=0.25, points= 100)
    data, funcval = func.FrankeFunction()
        
    learningrates = [1e-1, 1e-2, 1e-3, 1e-4]
    lambdas = [0]
    #lambdas= np.zeros(5)
    #lambdas[:4] = np.power(10.0,-1+-1*np.arange(4))

    MSE = np.zeros((len(learningrates),len(lambdas)))

    split_data = np.array_split(data,batches,axis=0)
    split_funcval = np.array_split(funcval,batches)

    i = 0
    for lr in learningrates:
        j = 0
        for param in lambdas:
            costfunc = Ridge(hyperpar=param)
            nn = FFNN(X_train = data, trainval = funcval,
                h_layors = 1, h_neurons = 30, categories = 1,
                CostFunc = costfunc, 
                h_actf = act_func.sigmoid,
                o_actf = act_func.identity,
                methode = "const", learningrate = lr)
            
            np.random.seed(1999) #ensures reproducibility
            for itera in range(epochs):
                for _ in range(batches):
                    rd_ind = np.random.randint(batches)
                    z,a = nn.FF(split_data[rd_ind])
                    nn.backpropagation(z,a,split_funcval[rd_ind])
                    nn.update_WandB()
            z,a = nn.FF()
            MSE[i,j] = costfunc.func(funcval,a[len(a)-1])
            j += 1
        i += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='1.3e')
    ax.set_xlabel("lambda")
    ax.set_ylabel("log$_{10}$(eta)")
    ax.set_xticklabels(lambdas)
    ax.set_yticklabels(np.log10(learningrates))
