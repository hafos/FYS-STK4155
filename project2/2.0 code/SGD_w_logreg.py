import numpy as np
from sklearn.linear_model import LogisticRegression

class StochGradDecent:
    """Class for performing Stochastic Gradient Decent methods on a given dataset"""
    def __init__(self, X_train = None, f_train = None, costfunc = None) -> None:
        """ 
        Constructor for generating an instance of the class.
        
        Arguments
        
        ---------
        X_train: array
            train data values (default: None)
        trainval: array
            train function values (default: None)
        costfunc: class
            If one wants to use gradient decent, a cost function and the derivative
            has to be provided (default: None)

        Errors
        ------
        TypeError:
            If no data is provided
            If no cost function is provided for gradient decent
        Index Error:
            Dimension of given starting beta is wrong
        """
        
        if X_train is None or f_train is None:
            raise TypeError("Class needs data as Input: <X_train> and <trainval> not provided")	
        if costfunc is None: 
            raise TypeError("Cost func is missing")
        
        self.X_train = X_train
        self.f_train = f_train
        self.costfunc= costfunc
        np.random.seed(1999)
        self.beta = np.random.randn(X_train.shape[1],1)
        
    def const(self, epochs = int(10e2), batches = 10, learningrate= 10e-3):
        """
        Stochastic Gradient Decent with a constant learningrate
        
        Arguments
        ---------
        epochs: int
            Number of epochs (default: 10e2)  
        batches: int
            Number of batches (default: 10)
        learningrate: float
            learningrate (default: 10e-3)
        """
        
        X_train = self.X_train.copy()
        f_train = self.trainval.copy()
        beta = self.beta.copy()
        
        X_train = np.array_split(X_train,batches,axis=0)
        f_train = np.array_split(f_train,batches)
        
        np.random.seed(1999) #ensures reproducibility
        for itera in range(epochs):
            for i in range(batches):
                rd_ind = np.random.randint(batches)
                costfunc = self.costfunc
                gradient = costfunc.derivative(f_train[rd_ind],X_train[rd_ind],beta)
                beta -= learningrate*gradient
        
        return(beta)
    
    def adaptive(self, epochs = int(10e2), batches = 10, t_0 = 10e-1,
                 t_1=1.0):
        """
        Stochastic Gradient Decent with a adaptive learningrate
        
        Arguments
        ---------
        epochs: int
            Number of epochs (default: 10e2)  
        batches: int
            Number of batches (default: 10)
        t_0: float
            parameter 1 (default: 10e-3)
        t_1: float
            parameter 2 (default: 1.0)
        """
        
        X_train = self.X_train.copy()
        f_train = self.f_train.copy()
        beta = self.beta.copy()
        
        X_train = np.array_split(X_train,batches,axis=0)
        f_train = np.array_split(f_train,batches)
        
        np.random.seed(1999) #ensures reproducibility
        for itera in range(epochs):
            for i in range(batches):
                rd_ind = np.random.randint(batches)
                costfunc = self.costfunc
                gradient = costfunc.derivative(f_train[rd_ind],X_train[rd_ind],beta)
                learningrate = t_0/(t_1+itera)
                beta -= learningrate*gradient
        
        return(beta)
        
    
    def momentum(self, epochs = int(10e2), batches = 10, learningrate = 10e-1, 
                 delta_momentum = 0.3):
        """
        Momentum based Stochastic Gradient Decent
        
        Arguments
        ---------
        epochs: int
            Number of epochs (default: 10e2)  
        batches: int
            Number of batches (default: 10)
        learningrate: float
            Starting learningrate (default: 10e-2)
        delta_momentum: float
            momentum parameter (default: 0.3)           
        """
        
        X_train = self.X_train.copy()
        f_train = self.f_train.copy()
        beta = self.beta.copy()
        
        change = 0
        
        X_train = np.array_split(X_train,batches,axis=0)
        f_train = np.array_split(f_train,batches)
        
        np.random.seed(1999) #ensures reproducibility
        for itera in range(epochs):
            for i in range(batches):
                rd_ind = np.random.randint(batches)
                costfunc = self.costfunc
                gradient = costfunc.derivative(f_train[rd_ind],X_train[rd_ind],beta)
                new_change = learningrate*gradient+delta_momentum*change
                beta -= new_change
                change = new_change
        
        return beta 
    
    def adagrad(self, epochs = int(10e2), batches = 10, learningrate= 10e-1, 
                momentum = False, delta_momentum = 0.3):
        """
        Stochastic Gradient Decent with ADAGRAD
        
        Arguments
        ---------
        epochs: int
            Number of epochs (default: 10e2)  
        batches: int
            Number of batches (default: 10)  
        learningrate: float
            Starting learningrate (default: 10e-1)
        momentum: boolean
            Choose if ADAGRAD is perfomed with or without momentum (default: False)
        delta_momentum: float
            momentum parameter (default: 0.3)
            
        Errors
        ---------
        TypeError: 
            if <momentum> is not bolean
        """
        
        X_train = self.X_train.copy()
        f_train = self.f_train.copy()
        beta = self.beta.copy()
        Giter = np.zeros((X_train.shape[1],X_train.shape[1]))
        delta  = 1e-8
        change = 0
        
        X_train = np.array_split(X_train,batches,axis=0)
        f_train = np.array_split(f_train,batches)
        
        np.random.seed(1999) #ensures reproducibility
        for itera in range(epochs):
            for i in range(batches):
                rd_ind = np.random.randint(batches)
                costfunc = self.costfunc
                gradient = costfunc.derivative(f_train[rd_ind],X_train[rd_ind],beta)
                Giter += gradient @ gradient.T
                coef = np.c_[learningrate/(delta+np.sqrt(np.diagonal(Giter)))]
                
                match momentum: 
                    case True: 
                        new_change = np.multiply(coef,gradient) + delta_momentum*change
                        change = new_change
                    case False: 
                        new_change = np.multiply(coef,gradient)
                    case _:
                        raise TypeError("<momentum> is a bolean variable")
                
                beta-=new_change
        
        return beta
    
    def rmsprop(self, epochs = int(10e2), batches = 10, learningrate= 10e-3, 
                t = 0.9):
        """
        Stochastic Gradient Decent with RMSprop
        
        Arguments
        ---------
        epochs: int
            Number of epochs (default: 10e2)  
        batches: int
            Number of batches (default: 10)  
        learningrate: float
            Starting learningrate (default: 10e-3)
        t: float
            averaging time of the second moment (default: 0.9)
        """
        
        X_train = self.X_train.copy()
        f_train = self.f_train.copy()
        beta = self.beta.copy()
        s = np.zeros((X_train.shape[1],1)) 
        delta  = 1e-8
        
        X_train = np.array_split(X_train,batches,axis=0)
        f_train = np.array_split(f_train,batches)
        np.random.seed(1999) #ensures reproducibility
        
        for itera in range(epochs):
            for i in range(batches):
                rd_ind = np.random.randint(batches)
                costfunc = self.costfunc
                gradient = costfunc.derivative(f_train[rd_ind],X_train[rd_ind],beta)
                s = t*s + (1-t)*np.power(gradient,2)
                coef = learningrate/np.sqrt(delta+np.sqrt(s))
                beta -= np.multiply(coef,gradient)
        
        return beta

    def adam(self, epochs = int(10e2), batches = 10, learningrate= 0.1, 
             t1 = 0.9, t2 = 0.99):
        """
        Stochastic Gradient Decent with ADAM
        
        Arguments
        ---------
        epochs: int
            Number of epochs (default: 10e2)  
        batches: int
            Number of batches (default: 10)    
        learningrate: float
            Starting learningrate (default: 10e-3)
        t1: float
            averaging time of the first moment (default: 0.9)
        t2: float
            averaging time of the second moment (defualt:0.99)
        """
        
        X_train = self.X_train.copy()
        f_train = self.trainval.copy()
        beta = self.beta.copy()
        m = np.zeros((X_train.shape[1],1))
        s = np.zeros((X_train.shape[1],1))
        delta  = 1e-8
        
        X_train = np.array_split(X_train,batches,axis=0)
        f_train = np.array_split(f_train,batches)
        
        np.random.seed(1999) #ensures reproducibility
        for itera in range(epochs):
            for i in range(batches):
                rd_ind = np.random.randint(batches)
                costfunc = self.costfunc
                gradient = costfunc.derivative(f_train[rd_ind],X_train[rd_ind],beta)
                m = t1 * m + (1-t1) * gradient
                m_hat = m / (1 - np.power(t1,itera+1))
                s = t2 * s + (1-t2) * np.power(gradient,2)
                s_hat = s / (1 - np.power(t2,itera+1))
                coef = learningrate/(delta+np.sqrt(s_hat))
                beta -= np.multiply(coef,m_hat)
        
        return beta

    def logistic_regression(self, epochs=int(1e2), batches=10, eta=0.1, regularization=0.1):
        X_train = self.X_train.copy()
        f_train = self.f_train.copy()
        f_train = f_train.reshape((len(f_train),))
        beta = np.random.randn(X_train.shape[1],)
        costfunc = self.costfunc

        # m = int(X_train.shape[0]/batches)

        # for i in range(epochs):
        #     for j in range(batches):
        #         rd_ind = np.random.randint(batches)
        m = int(X_train.shape[0]/batches)

        for epoch in range(epochs):

            for i in range(m):
                rd_ind = batches * np.random.randint(m)
                xi = X_train[rd_ind:rd_ind + batches]
                yi = f_train[rd_ind:rd_ind + batches]
                gradients = (np.squeeze(costfunc.derivative(xi @ beta))-yi) @ xi + regularization * beta
                beta -= eta*gradients

        return beta



        return beta

    def logistic_regression_sklearn(self):
        X_train = self.X_train.copy()
        f_train = self.trainval.copy()

        log_reg = LogisticRegression(max_iter=int(1e4))
        # to test accuracy run log_reg.score(X_train, f_train) after .fit method
        return log_reg.fit(X_train, f_train)
