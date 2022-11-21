"""
Date: 21.11.22 

@author: semyaat
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import time

import sys
sys.path.append('classes')

from Linear_Regression import LinearRegression
from Grad_Decent import GradDecent
from S_Grad_Decent import StochGradDecent
from gen_data import functions
from cost_act_func import CostOLS_beta




# def CostOLS(y,X,beta):
#     return 1/X.shape[0]*((y-X@beta).T@(y-X@beta))

def timing(): 
    dimension = 2
    order = 2

    func = functions(order = order, dimension=dimension, sigma=0.0, points= 100)
    costfunc = CostOLS_beta

    data, funcval = func.FrankeFunction()
    poly = PolynomialFeatures(degree=order)
    X = poly.fit_transform(data)


    # x, y = np.meshgrid(np.arange(0, 1, 1/points), np.arange(0, 1, 1/points))
    # z = np.concatenate(franke_function(x, y, sigma, noise), axis=None)
    # dataset = [x, y]

    X_train, X_test, z_train, z_test = train_test_split(X, funcval, test_size=0.2, random_state=1)

    reg = LinearRegression(X_train, z_train)
    gd = GradDecent(X_train, z_train, costfunc=costfunc)
    sd = StochGradDecent(X_train, z_train, costfunc=costfunc)

    beta_reg = reg.ols()
    start = time.time()
    beta_GD         = gd.const()
    beta_GD_mom     = gd.momentum()
    beta_GD_ada     = gd.adagrad()
    beta_GD_ada_mom = gd.adagrad(momentum = True,learningrate = 1)
    beta_GD_rms     = gd.rmsprop()
    beta_GD_adam    = gd.adam()
    end = time.time()

    print(f'Calc time GD: {end-start}')

    start = time.time()
    beta_SD         = sd.const()
    beta_SD_mom     = sd.momentum()
    beta_SD_ada     = sd.adagrad()
    beta_SD_ada_mom = sd.adagrad(momentum = True,learningrate = 1)
    beta_SD_rms     = sd.rmsprop()
    beta_SD_adam    = sd.adam()
    end = time.time()

    print(f'Calc time SGD: {end-start}')


def compare_time():
    """ Calculates and prints the time per epoch and time per operation for 
        GD and SGD for a constant learning rate. 
    """

    epochs = 200
    batches = 100
    lr = 0.1
    gd = GradDecent(X, funcval, costfunc=costfunc)
    sd = StochGradDecent(X, funcval, costfunc=costfunc)
    times = np.zeros(2)

    start = time.time()
    beta_gd = gd.const(iterations = epochs, learningrate = lr)
    end = time.time()
    times[0] = end-start

    start = time.time()
    beta_sd = sd.const(epochs = epochs, batches = batches, learningrate = lr)
    end = time.time()
    times[1] = end-start

    size = np.array([1, batches])
    with np.printoptions(formatter={'float': lambda x: format(x, '6.3e')}):
        print("The first list element corresponds to GD and the second one to SGD")
        print(f"time per epoch in s: \t \t {times/epochs}")
        print(f"time per operation in s: \t {times/(epochs*size)}")

def compare_GD_SGD(gd, sd): 
    """ Compares Gradient Descent vs Stochastic Gradient Descent 
        with fixed learning rates (eta) and no momentum, 
        using the Mean Square error. 

        Plots: 
            1) Compares plain GD and SGD for const batch sizes 
            2) Compares plain GD and SGD for const learning rates 
    """

    print ("Different learningrates for constant batchsize: ...", end = "")
    learningrates = [10e-2, 10e-3]
    epochs = np.arange(200)

    # Initialize and calculate the MSE 
    MSE_gd = np.zeros((len(learningrates),len(epochs)))
    MSE_sd = np.zeros((len(learningrates),len(epochs)))

    i = 0 
    for lr in learningrates:
        j = 0
        for it in epochs:
            beta_gd = gd.const(iterations = it, learningrate = lr)
            beta_sd = sd.const(epochs = it, batches = 4, learningrate = lr)
            MSE_gd[i,j] = costfunc.func(funcval, X, beta_gd)
            MSE_sd[i,j] = costfunc.func(funcval, X, beta_sd)
            j +=1
        i+=1

    # Plot 
    plt.figure("1")
    colors = ["green", "blue"]
    for i in range(len(learningrates)):
        plt.plot(epochs, MSE_gd[i,:], label = fr"GD with $\eta$ = {learningrates[i]}",
                color = colors[i])
        plt.plot(epochs, MSE_sd[i,:], label = fr"SGD with $\eta$ = {learningrates[i]}",
                linestyle = "dashed", color = colors[i])
        plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("epochs")
    plt.savefig("figures/MSE_SG_SGD_learning_rates.pdf")
    # plt.show()
    print("[DONE]")

    print ("Different batchsizes for constant learningrate: ...", end = "")
    epochs = np.arange(0,200)
    batchsizes = [1,4,16,64]

    MSE_gd = np.zeros(len(epochs))
    MSE_sd = np.zeros((len(batchsizes),len(epochs)))

    i = 0
    for it in epochs:
        j = 0
        for bs in batchsizes:
            beta_sd = sd.const(epochs = it, batches = bs, learningrate = 0.01)
            MSE_sd[j,i] = costfunc.func(funcval, X, beta_sd)
            j +=1
        i += 1
    plt.figure("2")
    for j in range(MSE_sd.shape[0]):
        plt.plot(epochs,MSE_sd[j,:], label = f"Number of batches = {batchsizes[j]}")
    plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("epochs")
    plt.savefig("figures/MSE_SG_SGD_batch_sizes.pdf")
    # plt.show()
    print("[DONE]")

def SGD_lr_batches(sd): 
    """ Generates headmap, where MSE is plotted for different 
        learning rates and number of batches 

    """

    epochs = 100
    learningrates = [1e-1, 1e-2, 1e-3, 1e-4]
    batchsizes = [1,2,4,8,16,32,64,128,256,512,1024]

    MSE = np.zeros((len(batchsizes),len(learningrates)))

    i = 0
    for bs in batchsizes:
        j = 0
        for lr in learningrates:
            beta = sd.const(epochs = epochs, batches = bs, learningrate = lr)
            MSE[i,j] = costfunc.func(funcval, X, beta)
            j += 1
        i += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='1.3e')
    ax.set_xlabel("log$_{10}$(eta)")
    ax.set_ylabel("batchsize")
    ax.set_xticklabels(np.log10(learningrates))
    ax.set_yticklabels(batchsizes)

    plt.savefig("figures/colormap_learningrate_batches_MSE")
    # plt.show() 


func = functions(dimension=2, sigma=0.25 ,points= 100)
costfunc = CostOLS_beta

data, funcval = func.FrankeFunction()

order = 6
poly = PolynomialFeatures(degree=order)
X = poly.fit_transform(data)


X_train, X_test, z_train, z_test = train_test_split(X, funcval, test_size=0.2, random_state=1)

gd = GradDecent(X_train, z_train, costfunc=costfunc)
sd = StochGradDecent(X_train, z_train, costfunc=costfunc)

# compare_time()
# compare_GD_SGD(gd, sd)
# SGD_lr_batches(sd)