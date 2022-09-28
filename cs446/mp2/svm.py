

import torch
import hw1_utils as utils
import matplotlib.pyplot as plt
'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 4 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
'''

# Problem 3
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    w = torch.zeros(X.shape[1] + 1,1) # (d + 1) 0
    ones = torch.ones(X.shape[0],1)  # n 1
    X2 = torch.cat((ones, X), 1)
    n = X.shape[0]
    for i in range(num_iter):
        w = w - lrate/n*torch.t(X2)@(X2@w-Y)
        

    return w
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    pass

def linear_normal(X, Y):
    w = torch.zeros(X.shape[1] + 1) # (d + 1) 0
    ones = torch.ones(X.shape[0],1)  # n 1
    X2 = torch.cat((ones, X), 1)
    w = torch.pinverse(X2)@Y

    return w
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    pass

def plot_linear():

    X, Y = utils.load_reg_data()
    w = linear_normal(X, Y)
    plt.scatter(X, Y)
    f_x = X@w
    plt.plot(X,f_x, color = 'red')
    plt.show()

    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''

# Problem 4
def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    w = torch.zeros(int(X.shape[1] + 1+X.shape[1] *(X.shape[1]+1)/2),1) # (d + 1) 0
    ones = torch.ones(X.shape[0],1)  # n 1
    col = torch.zeros(X.shape[0],1)
    X2 = torch.cat((ones, X), 1)
    n = X.shape[0]
    for i in range(X.shape[1]):
        index = i
        while index < X.shape[1]:
            for index2 in range(X.shape[0]):
                col[index2, 0] = X[index2, i]*X[index2, index]
            index += 1
            X2 = torch.cat((X2,col), 1)
    for i in range(num_iter):
        w -= lrate/n*torch.t(X2)@(X2@w-Y)
    return w

    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    pass

def poly_normal(X,Y):
    w = torch.zeros(int(X.shape[1] + 1+X.shape[1] *(X.shape[1]+1)/2),1) # (d + 1) 0
    ones = torch.ones(X.shape[0],1)  # n 1
    col = torch.zeros(X.shape[0],1)
    X2 = torch.cat((ones, X), 1)
    n = X.shape[0]
    for i in range(X.shape[1]):
        index = i
        while index < X.shape[1]:
            for index2 in range(X.shape[0]):
                col[index2, 0] = X[index2, i]*X[index2, index]
            index += 1
            X2 = torch.cat((X2,col), 1)
    return torch.pinverse(X2)@Y
    
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    pass

def plot_poly():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    pass

def poly_xor():
    '''
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    '''
    pass

# Problem 5
def logistic(X, Y, lrate=.01, num_iter=1000):
    w = torch.zeros(X.shape[1] + 1,1) # (d + 1) 0
    ones = torch.ones(X.shape[0],1)  # n 1
    X2 = torch.cat((ones, X), 1)
    n = X.shape[0]
    for i in range(num_iter):

        w += lrate/n*torch.sum((X2*Y*torch.exp(-(X2@w)*Y))/(1+torch.exp(-(X2@w)*Y)), 0).reshape(-1,1)
            
    return w
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    pass

def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    pass