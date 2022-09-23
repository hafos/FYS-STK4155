import numpy as np 

def DesignMatrix(x, y, n): 
    """ Defines the matrix X 

        Args:
            x 
            y
            n: Order of polynomial 

        Returns:
            X: Design matrix 

    """

    # Flatten array using ravel 
    if len(x.shape) > 1: 
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n + 1)*(n + 2)/2) # Number of elements in beta 
    X = np.ones((N, l))

    j = 1 
    # columns = [r'$x^0 y^0$']
    for i in range(1, n+1): 
        q = int(i + (i+1)/2)
        for k in range(i+1): 
            X[:,j] = x**(i-k) * y**k 
            # X[:,q+k] = x**(i-k) * y**k # Got an out of range error here 
            # columns.append(r'$x^{%i} y^{%i}$'%((i-k), k))
            j += 1
            # print("j", j)
            # print("q", q+k)

    # print(columns) # Print x, y, x², xy, y², etc 
    return X 

def get_beta(X, y): 
    """ Uses matrix inversion to find beta (y = X*beta + epsilon)

    """
    # Train model 

    # b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) # Evt dette? 

    b = np.linalg.pinv(X.T @ X) @ X.T @ y.ravel()
    # b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

    return b


def MSE(y, y_tilde): 
    """ Evaluates the mean square error given y_data and the predicted y_model 
        Args: 
            y:       the data 
            y_tilde: the model / fit for the data 

    """

    mse = 1/np.size(y) * np.sum((y - y_tilde)**2)
    n = np.size(y)
    return np.sum((y-y_tilde)**2)/n


def R2(y, y_tilde): 
    """ Evaluates the R² score given y_data and the predicted y_model 
        Args: 
            y:       the data 
            y_tilde: the model / fit for the data 
    
    """

    y_mean = 1/len(y) * np.sum(y)
    return 1 - np.sum((y - y_tilde)**2) / np.sum((y - np.mean(y_tilde))**2)

def RelErr(y, y_tilde): 
    return abs((y-y_tilde)/y)