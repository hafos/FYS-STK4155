import matplotlib.pyplot as plt
import numpy as np 
import scipy as sp
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import resample


from functions import * 

n_boot = 100 
k = 10 

class LinearRegression(): 
    def __init__(self, z, order, X=None): 
        """ 
        Performs an regression analysis in x and y up to fifth order 

        args: 
            order: Maximum order 
            X: Design Matrix 

        """

        self.z = z 
        self.X = X 
        self.order = order 

        ## Initializing arrays 
        self.variables = [x, y] # Bringing variables on the right form 
        print(np.shape(self.variables))

        # Dim of array needed is given Complete homogeneous symmetric polynomial 
        numbofterms = sp.special.comb(len(self.variables) + order, order, exact=True)
        self.beta = np.full((order, numbofterms), np.nan)

        #Error arrays
        self.MSE_train = np.zeros(order)
        self.MSE_test  = np.zeros(order)
        self.R_two_train = np.zeros(order)
        self.R_two_test  = np.zeros(order)
        self.BIAS  = np.zeros(order)
        self.var   = np.zeros(order)

        ## Full Design Matrix
        if self.X == None: self.X = DM(self.variables, order)

    def TTS(self, ratio=0.2, scale=False): 
        ## Split data into trainning and testing
        A_train, A_test, z_train, z_test = train_test_split(self.X, self.z, test_size=ratio)
        if scale == True: 
            A_test -= np.mean(A_train,axis=0)
            A_train -= np.mean(A_train,axis=0)
            z_scale = np.mean(z_train)
            z_train -= z_scale
        return A_train, A_test, z_train, z_test


    def OLS(self, bootstrap=False, k_fold=False): 
        ## OLS regression analysis 

        # Initialize 
        A_train, A_test, z_train, z_test = self.TTS() # XXX WE ONLY SCALE FOR RIDGE (remember to write why)
        beta = self.beta 

        train_MSE = self.MSE_train; test_MSE = self.MSE_test
        train_R2 = self.R_two_train; test_R2 = self.R_two_test
        BIAS = self.BIAS
        var = self.var

        ytilde_test = np.empty((len(z_test), n_boot+1)) # bootstrap 
        

        for i in range(1, self.order+1):
            # XXX what is this ?? 
            currentnot = sp.special.comb(len(self.variables) + i, i, exact=True)
            ATrCur = A_train[:,0:currentnot]
            ATeCur = A_test[:,0:currentnot]
            
            # Matrix inversion to find beta 
            beta[i-1][:currentnot] = np.linalg.pinv(ATrCur.T @ ATrCur) @ ATrCur.T @ z_train # XXX pinv or inv? 
            
            if bootstrap==True: 
                # Calculate ytilde 
                ytilde_train     = ATrCur @ beta[i-1][~np.isnan(beta[i-1])] # fitted z 
                ytilde_test[:,0] = ATeCur @ beta[i-1][~np.isnan(beta[i-1])] # predicted z 
                # Perform the bootstrap on the test 
                for j in range(n_boot+1):
                    A_res, z_res = resample(ATrCur, z_train)
                    beta[i-1][:currentnot] = np.linalg.pinv(A_res.T @ A_res) @ A_res.T @ z_res
                    ytilde_test[:,j] = ATeCur @ beta[i-1][~np.isnan(beta[i-1])] 
                
            else: 
                # Calculate ytilde 
                ytilde_train = ATrCur @ beta[i-1][~np.isnan(beta[i-1])] # fitted z 
                ytilde_test  = ATeCur @ beta[i-1][~np.isnan(beta[i-1])] # predicted z 

        # return ytilde_train, ytilde_test, z_train, z_test

            ## Calcuate the error ## 

            if bootstrap==True: 
                train_MSE[i-1] = 1/len(z_train)*np.sum(np.power(z_train - ytilde_train, 2))
                test_MSE[i-1] = np.mean( np.mean((z_test.reshape(-1, 1) - ytilde_test)**2, axis=1, keepdims=True) )

                #Calculate BIAS & variance
                BIAS[i-1] = np.mean( (z_test.reshape(-1, 1) - np.mean(ytilde_test, axis=1, keepdims=True))**2 )
                var[i-1] = np.mean( np.var(ytilde_test, axis=1, keepdims=True) )
                print(f"MSE - BIAS + var = {(test_MSE[i-1] - (BIAS[i-1] + var[i-1])):.3e}")

            else: 
                train_MSE[i-1] = get_MSE(z_train, ytilde_train)
                train_R2[i-1] = R2(z_train, ytilde_train)

                test_MSE[i-1] = get_MSE(z_test, ytilde_test)
                test_R2[i-1] = R2(z_test, ytilde_test)

        self.beta_OLS = beta 
    
        return train_MSE, test_MSE, train_R2, test_R2, BIAS, var

    def error_analysis(self):
        # Create arrays 
        train_MSE = self.MSE_train; test_MSE = self.MSE_test
        train_R2 = self.R_two_train; test_R2 = self.R_two_test

        for n in range(1, self.order+1): 
            print(f"\nOrder {n}")
            ytilde_train, ytilde_test, z_train, z_test = self.OLS()

            # MSE 
            train_MSE = get_MSE(z_train.ravel(), ytilde_train.ravel())
            test_MSE  = get_MSE(z_test.ravel(), ytilde_test.ravel())
            # R2
            train_R2  = R2(z_train.ravel(), ytilde_train.ravel())
            test_R2   = R2(z_test.ravel(), ytilde_test.ravel())

        return train_MSE, test_MSE, train_R2, test_R2


    def ridge(self, scale=False): 
        #Ridge Stuff
        lambdas = np.zeros(9)
        lambdas[:8] = np.power(10.0,2-np.arange(8))

        #bootstrap
        n_bootstrap = 100
        ##Cross_val
        k= 10
        rs = KFold(n_splits=k,shuffle=True, random_state=1)

        # Initialize Error arrays
        MSE_bs = np.zeros((len(lambdas), self.order))
        scores_KFold = np.zeros(k)
        MSE_KFOLD = np.zeros((len(lambdas), self.order))

        for i in range(1, self.order+1):
            currentnot = sp.special.comb(len(self.variables) + i,i,exact=True)
            A_curr = self.X[:,0:currentnot]
            k = 0
            for ridge_par in lambdas: 
                j=0
                for train_index, test_index in rs.split(A_curr):
                    A_train = A_curr[train_index]
                    z_train = z[train_index]
                    A_test = A_curr[test_index]
                    z_test = z[test_index]
                    ##rescale
                    if scale == True:
                        A_test  -= np.mean(A_train,axis=0)
                        A_train -= np.mean(A_train,axis=0)
                        z_scale = np.mean(z_train)
                        z_train -= z_scale
                    ##
                    beta = np.linalg.pinv(A_train.T @ A_train + ridge_par*np.identity(A_train.T.shape[0])) 
                    beta = beta @ A_train.T @ z_train
                    f_approx_test= A_test @ beta + z_scale
                    scores_KFold[j] = np.sum((f_approx_test - z_test)**2)/np.size(f_approx_test)
                    j +=1
                MSE_KFOLD[k][i-1] = np.mean(scores_KFold)
                k += 1
        min_Kfold_ind = divmod(MSE_KFOLD.argmin(), MSE_KFOLD.shape[1])

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(MSE_KFOLD.T, annot=True, ax=ax, cmap="viridis", cbar_kws={'label': 'Accuracy'},fmt='.1e')
        ax.set_title("Test Accuracy KFOLD 1")
        ax.set_ylabel("order")
        ax.set_xlabel("log$_{10}(\lambda)$")
        ax.set_xticklabels(np.log10(lambdas,out=np.zeros_like(lambdas), where=(lambdas!=0)))
        ax.set_yticklabels(range(1, self.order+1))
        ax.add_patch(plt.Rectangle((min_Kfold_ind[0], min_Kfold_ind[1]), 1, 1, fc='none', ec='red', lw=2, clip_on=False))    





    
    def lasso(self): 
        pass 


    def k_fold(self, k=10): 
        n_boot = 100

        # Error arrays 
        MSE_bs = np.zeros(order)
        scores_KFold = np.zeros(k)
        MSE_KFOLD = np.zeros(order)
        BIAS = np.zeros(order)
        var = np.zeros(order)

        # Split data into test & train
        rs = KFold(n_splits=k, shuffle=True)
        for i in range(1, self.order+1):
            currentnot = sp.special.comb(len(self.variables) + i,i,exact=True)    
            j=0
            A_curr = self.X[:,0:currentnot]
            for train_index, test_index in rs.split(A_curr):
                A_train = A_curr[train_index]
                z_train = z[train_index]
                A_test = A_curr[test_index]
                z_test = z[test_index]
                beta = np.linalg.pinv(A_train.T @ A_train) @ A_train.T @ z_train
                f_approx_test= A_test @ beta
                scores_KFold[j] = np.sum((f_approx_test - z_test)**2)/np.size(f_approx_test)
                j +=1
            MSE_KFOLD[i-1] = np.mean(scores_KFold)
            A_train, A_test, z_train, z_test = train_test_split(A_curr, z, test_size=0.2)
            beta = np.linalg.inv(A_train.T @ A_train) @ A_train.T @ z_train
            f_approx_test = np.empty((len(z_test), n_boot+1))
            f_approx_test[:,0] = A_test @ beta
            for j in range(n_boot+1):
                A_res, z_res = resample(A_train, z_train)
                beta = np.linalg.inv(A_train.T @ A_train) @ A_train.T @ z_train
                f_approx_test[:,j] = A_test @ beta
            MSE_bs[i-1] = np.mean( np.mean((z_test.reshape(-1, 1) - f_approx_test)**2, axis=1, keepdims=True) )
        return MSE_bs, MSE_KFOLD

    ### Plots 
    def ex_b(self): 
        p_arr = np.arange(1, self.order+1, 1.0)
        MSE_train, MSE_test, R2_train, R2_test, BIAS, var = self.OLS()
        # MSE_train, MSE_test, R2_train, R2_test = self.error_analysis()


        fig, ax = plt.subplots(nrows=2, sharex=True)
        # Data 
        ax[0].plot(p_arr, MSE_train, label="Train MSE", color="orange")
        ax[0].scatter(p_arr, MSE_train, color="orange", s=15)
        ax[1].plot(p_arr, R2_train, label="Train R2", color="blue")
        ax[1].scatter(p_arr, R2_train, color="blue", s=15)
        # Predicted 
        # ax[0].plot(p_arr, MSE_test, label="Test MSE")
        ax[0].scatter(p_arr, MSE_test, label="Test MSE", color="orange", s=15)
        # ax[1].plot(p_arr, R2_test, label="Test R2")
        ax[1].scatter(p_arr, R2_test, label="Test R2", color="blue", s=15)
        # ax[0].set_yscale('log')
        ax[0].legend()
        ax[1].legend()
        fig.tight_layout()
        plt.show()

        for i in range(0,len(self.beta_OLS[0])):
            plt.scatter(p_arr, self.beta_OLS[0:,i], s=15)
        plt.show()
    
    def ex_c(self): 
        ## Reproduce fig 2.11 of Hastie, Tibshirani, and Friedman
        p_arr = np.arange(1, self.order+1, 1.0)
        MSE_train, MSE_test, R2_train, R2_test, BIAS, var = self.OLS(bootstrap=True)

        plt.scatter(p_arr, MSE_train, label='train', color='orange', s=15)
        plt.scatter(p_arr, MSE_test, label='test', color='blue', s=15)
        plt.legend(loc = "upper center")
        plt.show()

        plt.scatter(p_arr, BIAS,     label='BIAS',     color='orange', s=15)
        plt.scatter(p_arr, MSE_test, label='MSE Test', color='blue', s=15)
        plt.scatter(p_arr, var,      label='Variance', color='red', s=15)     
        plt.legend(loc = "upper center")
        plt.show()
    
    def ex_d(self, no): 
        p_arr = np.arange(1, self.order+1, 1.0)

        if no == 1: 
            MSE_bs, MSE_KFOLD = self.k_fold()

            plt.scatter(p_arr, MSE_bs, label='MSE_bs', color='orange', s=15)
            plt.scatter(p_arr, MSE_KFOLD, label='MSE_KFOLD', color='blue', s=15)    
            plt.legend()
            plt.show()
        if no == 2: 
            # Compare cross validation with bootstrap (OLS) 
            for i in range(5,10+1):
                MSE_bs, MSE_KFOLD = self.k_fold(i)
                plt.scatter(p_arr, MSE_KFOLD, label=str(i), s=15)
            plt.legend()
            plt.show()

    def ex_e(self): 
        self.ridge(True)

if __name__ == "__main__":
    np.random.seed(2018)
    
    # Exercise b)
    ## Generate datapoints 
    datapoints = 100
    sigma = 0.1
    order = 5

    ## Make data.
    # x = np.random.normal(0,1,datapoints)
    # y = np.random.normal(0,1,datapoints)
    x = np.random.rand(datapoints)
    y = np.random.rand(datapoints)

    x, y = np.meshgrid(x,y)
    z = np.concatenate(FrankeFunction(x, y, sigma), axis=None)  

    LR = LinearRegression(z, order)
    # LR.ex_b()
    
    # Exercise c)
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    sigma = 0.05
    x, y = np.meshgrid(x,y)
    z = np.concatenate(FrankeFunction(x, y, sigma), axis=None)
    order = 15

    LR = LinearRegression(z, order)
    # LR.ex_c()
    
    # Exercise d) 1
    x = np.arange(0, 1, 0.025)
    y = np.arange(0, 1, 0.025)
    sigma = 0.25
    x, y = np.meshgrid(x,y)
    z = np.concatenate(FrankeFunction(x, y, sigma), axis=None)
    order = 10

    LR = LinearRegression(z, order)
    # LR.ex_d(no=1)
    
    # Exercise d) 2 
    order = 15 
    LR = LinearRegression(z, order)
    # LR.ex_d(no=2)

    # Exercise e) Ridge regression 

    # Write code for the Ridge method
    # Add bootstrap analysis 
    # Add cross validation 
    # Compare and analyze results (plot some stuff)
    # Study dependence on lambda 
    LR = LinearRegression(z, order)
    LR.ex_e()

    # Study bias-variance trade-off as function of various values of lambda (bootstrap)
    
