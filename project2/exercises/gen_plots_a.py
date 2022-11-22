"""
Date: 21.11.22 

@author: semyaat
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import numpy as np 
import time

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

import sys
sys.path.append('classes')

from Linear_Regression import LinearRegression
from Grad_Decent import GradDecent
from S_Grad_Decent import StochGradDecent
from gen_data import functions
from cost_act_func import CostOLS_beta, Ridge_beta

def timing(X, funcval): 
    reg = LinearRegression(X, funcval)
    gd = GradDecent(X, funcval, costfunc=costfunc)
    sd = StochGradDecent(X, funcval, costfunc=costfunc)

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


def compare_time(X, funcval):
    """ Calculates and prints the time per epoch and time per operation for 
        GD and SGD for a constant learning rate. 
    """

    epochs = 200
    batches = 100
    lr = 0.1
    # lr = 0.01
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

def compare_GD_SGD(X, funcval): 
    """ Compares Gradient Descent vs Stochastic Gradient Descent 
        with fixed learning rates (eta) and no momentum, 
        using the Mean Square error. 

        Plots: 
            1) Compares plain GD and SGD for const batch sizes 
            2) Compares plain GD and SGD for const learning rates 
    """
    gd = GradDecent(X, funcval, costfunc=costfunc)
    sd = StochGradDecent(X, funcval, costfunc=costfunc)


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
    plt.tight_layout()
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
    plt.tight_layout()
    plt.ylabel("MSE")
    plt.xlabel("epochs")

    plt.savefig("figures/MSE_SG_SGD_batch_sizes.pdf")
    # plt.show()
    print("[DONE]")

def SGD_lr_batches(X, funcval): 
    """ Generates headmap, where MSE is plotted for different 
        learning rates and number of batches 

    """
    sd = StochGradDecent(X, funcval, costfunc=costfunc)

    epochs = 100
    # learningrates = [1e-1, 1e-2, 1e-3, 1e-4]
    learningrates = [0.1, 0.01, 0.001, 0.0001]
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
    # ax.set_xlabel("log$_{10}$(eta)")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("batch size")
    # ax.set_xticklabels(np.log10(learningrates))
    ax.set_xticklabels(learningrates)
    ax.set_yticklabels(batchsizes)
    plt.setp(ax.get_yticklabels(), rotation=30)
    plt.tight_layout()

    plt.savefig("figures/colormap_learningrate_batches_MSE.pdf")
    # plt.show() 

def SGD_ridge(X, funcval): 
    """ 
    """
    reg = LinearRegression(X, funcval)

    epochs = 150
    batches = 32
    
    # learningrates = [1e-1, 1e-2, 1e-3, 1e-4]
    learningrates = [0.1, 0.01, 0.001, 0.0001]
    lambdas= np.zeros(5)
    lambdas[:4] = np.power(10.0,-1+-1*np.arange(4))

    MSE = np.zeros((len(learningrates),len(lambdas)))

    i = 0
    for lr in learningrates:
        j = 0
        for params in lambdas:
            costfunc = Ridge_beta(hyperpar = params)
            sd = StochGradDecent(X, funcval, costfunc=costfunc)
            beta = sd.const(epochs = epochs, batches = batches, learningrate = lr)
            MSE[i,j] = costfunc.func(funcval, X, beta)
            j += 1
        i += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(MSE, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'MSE'}, fmt='1.3e')
    ax.set_xlabel(r"$\lambda$")
    # ax.set_ylabel("log$_{10}$(eta)")
    ax.set_ylabel(r"$\eta$")
    ax.set_xticklabels(lambdas)
    # ax.set_yticklabels(np.log10(learningrates))
    ax.set_yticklabels(learningrates)
    plt.tight_layout()

    plt.savefig("figures/colormap_lambda_learningrate_MSE.pdf")
    # plt.show()

def compare_GD_SGD_all(X, funcval): 
    """ Compares Gradient Descent vs Stochastic Gradient Descent 
        with fixed learning rates (eta) and no momentum, 
        using the Mean Square error. 

        Plots: 
            1) Compares plain GD and SGD for const batch sizes 
            2) Compares plain GD and SGD for const learning rates 
    """
    gd = GradDecent(X, funcval, costfunc=costfunc)
    sd = StochGradDecent(X, funcval, costfunc=costfunc)


    print ("Different learningrates for constant batchsize: ...", end = "")
    lr = 10e-2
    epochs = np.arange(100)

    # lr = np.arange(1e-5,0)
    # epochs = 100 

    # Initialize and calculate the MSE 
    MSE_gd = np.zeros(len(epochs))
    MSE_gd_mom = np.zeros(len(epochs))
    MSE_gd_ada = np.zeros(len(epochs))
    MSE_gd_ada_mom = np.zeros(len(epochs))
    MSE_gd_rms = np.zeros(len(epochs))
    MSE_gd_adam = np.zeros(len(epochs))

    MSE_sd = np.zeros(len(epochs))
    MSE_sd_mom = np.zeros(len(epochs))
    MSE_sd_ada = np.zeros(len(epochs))
    MSE_sd_ada_mom = np.zeros(len(epochs))
    MSE_sd_rms = np.zeros(len(epochs))
    MSE_sd_adam = np.zeros(len(epochs))

    j = 0
    for it in epochs:
        # Initialize betas 
        beta_GD         = gd.const(iterations = it, learningrate = lr)    # Plain GD 
        beta_GD_mom     = gd.momentum(iterations = it, learningrate = lr) # GD with momentum 
        beta_GD_ada     = gd.adagrad(iterations = it, learningrate = lr)
        beta_GD_ada_mom = gd.adagrad(momentum = True, iterations = it, learningrate = lr)
        beta_GD_rms     = gd.rmsprop(iterations = it, learningrate = lr)
        beta_GD_adam    = gd.adam(iterations = it, learningrate = lr)

        beta_SD         = sd.const(epochs = it, batches = 4, learningrate = lr)    # Plain SGD 
        beta_SD_mom     = sd.momentum(epochs = it, batches = 4, learningrate = lr) # SGD with momentum 
        beta_SD_ada     = sd.adagrad(epochs = it, batches = 4, learningrate = lr)
        beta_SD_ada_mom = sd.adagrad(momentum = True, epochs = it, batches = 4, learningrate = lr)
        beta_SD_rms     = sd.rmsprop(epochs = it, batches = 4, learningrate = lr)
        beta_SD_adam    = sd.adam(epochs = it, batches = 4, learningrate = lr)

        # Calculate MSE 
        MSE_gd[j]         = costfunc.func(funcval, X, beta_GD)
        MSE_gd_mom[j]     = costfunc.func(funcval, X, beta_GD_mom)
        MSE_gd_ada[j]     = costfunc.func(funcval, X, beta_GD_ada)
        MSE_gd_ada_mom[j] = costfunc.func(funcval, X, beta_GD_ada_mom)
        MSE_gd_rms[j]     = costfunc.func(funcval, X, beta_GD_rms)
        MSE_gd_adam[j]    = costfunc.func(funcval, X, beta_GD_adam)

        MSE_sd[j]         = costfunc.func(funcval, X, beta_SD)
        MSE_sd_mom[j]     = costfunc.func(funcval, X, beta_SD_mom)
        MSE_sd_ada[j]     = costfunc.func(funcval, X, beta_SD_ada)
        MSE_sd_ada_mom[j] = costfunc.func(funcval, X, beta_SD_ada_mom)
        MSE_sd_rms[j]     = costfunc.func(funcval, X, beta_SD_rms)
        MSE_sd_adam[j]    = costfunc.func(funcval, X, beta_SD_adam)

        j +=1

    # Plot 
    plt.figure("1")
    colors = ["green", "blue"]
    plt.plot(epochs, MSE_gd, label = fr"GD", color="red")
    plt.plot(epochs, MSE_gd_mom, label = fr"GD with momentum", color="orange")
    plt.plot(epochs, MSE_gd_ada, label = fr"GD Adagrad", color="purple")
    plt.plot(epochs, MSE_gd_ada_mom, label = fr"GD Adagrad, momentum", color="blue")
    plt.plot(epochs, MSE_gd_rms, label = fr"GD RMS", color="black")
    plt.plot(epochs, MSE_gd_adam, label = fr"GD ADAM", color="lightgreen")

    plt.plot(epochs, MSE_sd, label = fr"SGD", linestyle = "dashed", color="red")
    plt.plot(epochs, MSE_sd_mom, label = fr"SGD with momentum", linestyle = "dashed", color="orange")
    plt.plot(epochs, MSE_sd_ada, label = fr"SGD Adagrad", linestyle = "dashed", color="purple")
    plt.plot(epochs, MSE_sd_ada_mom, label = fr"SGD Adagrad, momentum", linestyle = "dashed", color="blue")
    plt.plot(epochs, MSE_sd_rms, label = fr"SGD RMS", linestyle = "dashed", color="black")
    plt.plot(epochs, MSE_sd_adam, label = fr"SGD Adam", linestyle = "dashed", color="lightgreen")

    plt.legend()
    plt.tight_layout()
    plt.ylabel("MSE")
    plt.xlabel("epochs")
    plt.savefig("figures/MSE_SG_SGD_learning_rates_ALL.pdf")
    # plt.show()
    print("[DONE]")



## Initialize 
func = functions(dimension=2, sigma=0.25 , points=100)
costfunc = CostOLS_beta

data, funcval = func.FrankeFunction()

order = 6
poly = PolynomialFeatures(degree=order)
X = poly.fit_transform(data)

X_train, X_test, z_train, z_test = train_test_split(X, funcval, test_size=0.2, random_state=1)

## Run plots 
# compare_time(X_train, z_train)
# compare_GD_SGD(X_train, z_train)
# SGD_lr_batches(X_train, z_train)
# SGD_ridge(X_train, z_train)
compare_GD_SGD_all(X, funcval)