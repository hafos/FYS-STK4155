import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns

from sklearn.model_selection import train_test_split

#Import all needed functions from different py file
from funcs import functions as fnc

##global Parameters

np.random.seed(2018)

#IF sigma is zero there is no noise in the Model
sigma = 0.1
#Number of x & y points, total amount of datapoints is this squared
points = 15
#Programm will fit a polnyomial between min and max order
min_order = 1
max_order = 10

##create data and designe matrix 

x,y,fval = fnc.makedata(points,sigma,fnc.FrankeFunction)
variables=[x,y]
A = fnc.DM(variables,max_order)
fnc.rescale = False
#split data into test & train
A_train, A_test, f_train, f_test = train_test_split(A, fval, test_size=0.2, random_state = 1)

#different number of bootstraps
bootstraps = np.arange(0,100,10)
bootstraps[0] = 1


##create all needed arrays 

MSE_bs = np.zeros((len(bootstraps),max_order-min_order+1))


k=0
for bootstrap in bootstraps:
    i = 0
    for order in range(min_order,max_order+1):
        #current number of tearms also via complete homogeneous symmetric polynomials
        currentnot = sp.special.comb(len(variables) + i,i,exact=True)
        #select only the terms of the full desinge matrix needed for the current order
        ATrCur = A_train[:,0:currentnot]
        ATeCur = A_test[:,0:currentnot]
        MSE_bs[k,i] = fnc.bootstrap(bootstrap,ATrCur,ATeCur,f_train,f_test,fnc.OLS)[0]
        i +=1
    k += 1
    
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(MSE_bs.T, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'},fmt='.1e')
ax.set_title("Different number of bootstraps")
ax.set_ylabel("order")
ax.set_xlabel("bootstraps")
ax.set_xticklabels(bootstraps)
ax.set_yticklabels(np.arange(min_order,max_order+1,1))


