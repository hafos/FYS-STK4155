import numpy as np 

def DesignMatrix(x, y, p): 
    """ Defines the matrix X 

        Args:
            x 
            y
            p: Order of polynomial 

        Returns:
            X: Design matrix 

    """

    # Flatten array using ravel ? XXX
    # if len(x.shape) > 1: 
    x = np.ravel(x)
    y = np.ravel(y)

    N = len(x)
    a = int((p + 1)*(p + 2)/2) # Number of elements in beta 
    X = np.ones((N,a))

    j = 1 
    # columns = [r'$x^0 y^0$']
    for i in range(1, p+1): 
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
    """ Evaluates the mean square error given y and the predicted y 

    """

    mse = 1/np.size(y) * np.sum((y - y_tilde)**2)

    return mse 

def R2(y, y_tilde): 
    """ Evaluates R² 
    
    """

    y_mean = 1/len(y) * np.sum(y)
    r = 1 - np.sum((y - y_tilde)**2) / np.sum((y - np.mean(y_tilde))**2)
    return r 
