import numpy as np

# Constants
SIGMOID = "sigmoid"
MEAN_SQUARE_ERROR = "mse"
SOFT_MAX = "softmax"
RE_LU = "relu"
CROSS_ENTROPY = "cross_entropy"
TAN_H = "tanh"

# Mean Square Error
def mse(y_true,y_pred):
    total = 0
    n = len(y_true)
    for y,y1 in zip(y_true,y_pred):
        total+=(y-y1)**2
    avg = total/n
    return avg

# Sigmoid Activation Fn()
def sigmoid(n):
    """
    Sigmoid function applied element wise F(X) = [f(x0),f(x1),f(x2)]
    Used to convert values in the range of 0 to 1

    Attributes
    - n(np.Array | int)

    Return
    np.Array | int
    """
    return 1.0 / (1 + np.exp(-n))

#Softmax Output Fn()
def softmax(X):
    return np.exp(X) / np.sum(np.exp(X),axis=1, keepdims=True)

# Log Loss or binary Cross Entropy
def crossEntropy(Y):
    total = 0
    n = len(Y)
    for y in Y:
        total+=(y*np.log(y)+(1-y)*np.log((1-y)))
    return (-1*total)/n

# tanh
def tanh(z):
    # return np.tanh(z)
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

#Relu
def reLu(z):
    return np.maximum(z,0.0)

activation = {
    SIGMOID :sigmoid,
    SOFT_MAX :softmax,
    RE_LU :reLu,
    CROSS_ENTROPY :crossEntropy,
    TAN_H :tanh,
}
# MEAN_SQUARE_ERROR :mse,
