import matplotlib.pyplot as plt
import numpy as np 
import scipy as sp

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from functions import * 



class LinearRegression(): 
    def __init__(self, z, order, X=None): 
        """ 
        Performs an regression analysis in x and y up to fifth order 

        args: 
            order: Maximum order 
            X: Design Matrix 

        """
        
        self.X = X 
        self.order = order 

        

        ## Initializing arrays 

        self.variables = [x, y] # Bringing variables on the right form 

        # Dim of array needed is given Complete homogeneous symmetric polynomial 
        numbofterms = sp.special.comb(len(self.variables) + order, order, exact=True)
        self.beta = np.full((order, numbofterms), np.nan)

        #Error arrays
        self.MSE   = np.zeros(order)
        self.R_two = np.zeros(order)

        ## Full Design Matrix
        if X == None: X = DM(self.variables, order)

        ## Split data into trainning and testing
        ratio = 0.2 
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=ratio)

        self.X_train = X_train 
        self.X_test  = X_test 
        self.z_train = z_train 
        self.z_test  = z_test 

    def OLS(self): # XXX WE ONLY SCALE FOR RIDGE (remember to write why)
        ## OLS regression analysis 

        # Initialize 
        X_train = self.X_train; X_test = self.X_test
        z_train = self.z_train; z_test = self.z_test
        train_MSE = self.MSE; test_MSE = self.MSE 
        train_R2 = self.R_two; test_R2 = self.R_two
        beta = self.beta 

        for i in range(1, self.order+1):
            currentnot = sp.special.comb(len(self.variables) + i,i,exact=True)
            ATrCur = X_train[:,0:currentnot]
            ATeCur = X_test[:,0:currentnot]

            # Matrix inversion to find beta 
            beta[i-1][:currentnot] = np.linalg.pinv(ATrCur.T @ ATrCur) @ ATrCur.T @ z_train
            ytilde_train = ATrCur @ beta[i-1][~np.isnan(beta[i-1])] # fitted z 
            ytilde_test  = ATeCur @ beta[i-1][~np.isnan(beta[i-1])] # predicted z 
            
            # Calcuate the error
            train_MSE[i-1] = 1/len(z_train)*np.sum(np.power(z_train - ytilde_train, 2))
            train_R2[i-1] = R2(z_train, ytilde_train)

            test_MSE[i-1] = 1/len(z_test)*np.sum(np.power(z_test - ytilde_test, 2))
            test_R2[i-1] = R2(z_test, ytilde_test)
        
        return train_MSE, test_MSE, train_R2, test_R2

    def plot_OLS(self): 
        p_arr = np.arange(1, order+1, 1.0)
        train_MSE, test_MSE, train_R2, test_R2 = self.OLS()

        
        fig, ax = plt.subplots(nrows=2, sharex=True)
        # Data 
        ax[0].plot(p_arr, train_MSE, label="Train MSE", color="orange")
        ax[0].scatter(p_arr, train_MSE, color="orange", s=15)
        ax[1].plot(p_arr, train_R2, label="Train R2", color="blue")
        ax[1].scatter(p_arr, train_R2, color="blue", s=15)
        # Predicted 
        # ax[0].plot(p_arr, test_MSE, label="Test MSE")
        ax[0].scatter(p_arr, test_MSE, label="Test MSE", color="orange", s=15)
        # ax[1].plot(p_arr, test_R2, label="Test R2")
        ax[1].scatter(p_arr, test_R2, label="Test R2", color="blue", s=15)

        # ax[0].set_yscale('log')
        ax[0].legend()
        ax[1].legend()
        fig.tight_layout()
        plt.show()


    def ridge(self): 
        pass
    
    def lasso(self): 
        pass 


if __name__ == "__main__":
    # Generate datapoints 

    datapoints = 100
    sigma = 0.1
    # Make data.
    x = np.random.normal(0,1,datapoints)
    y = np.random.normal(0,1,datapoints)

    x, y = np.meshgrid(x,y)
    z = np.concatenate(FrankeFunction(x, y, sigma),axis=None)

    order = 5

    LR = LinearRegression(z, order)

    # ex 2 
    LR.plot_OLS()
