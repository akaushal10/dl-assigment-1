import numpy as np

# Constants
SIGMOID_KEY = "sigmoid"
TANH_KEY = "tanh"
RELU_KEY = "ReLU"
XAVIER_KEY = "Xavier"
RANDOM_KEY = "random"
HE_KEY = "HE"
SGD_KEY="sgd"
MGD_KEY="momentum"
NAG_KEY="nag"
RMSPROP_KEY="rmsprop"
ADAM_KEY="adam"
NADAM_KEY="nadam"

CROSS_ENTROPY_KEY = 'cross_entropy'
MEAN_SQUARE_KEY = 'mean_squared_error'

FASHION_MNIST_DATASET_KEY = 'fashion_mnist'
MNIST_DATASET_KEY = 'mnist'

def sigmoid(z):
    # z = np.clip(z,500,-500)
    return 1.0 / (1 + np.exp(-(z)))


def tanh(z):
    return np.tanh(z)


def sin(z):
    return np.sin(z)


def reLu(z):
    return (z>0)*(z) + ((z<0)*(z)*0.01)
    #return np.maximum(z,0)
    #return np.where(z<0, 0.01*z, z)

def softmax(Z):
    # Z = np.clip(Z,500,-500)
    Z -= np.max(Z)
    # Compute softmax
    exp_Z = np.exp(Z)
    softmax_output = exp_Z / np.sum(exp_Z)
    return softmax_output


def del_sigmoid(z):
    # z = np.clip(z,500,-500)
    return  (1.0 / (1 + np.exp(-(z))))*(1 -  1.0 / (1 + np.exp(-(z))))

def del_tanh(z):
    return 1 - np.tanh(z) ** 2


def del_reLu(z):
    return (z>0)*np.ones(z.shape) + (z<0)*(0.01*np.ones(z.shape) )

def Xavier_initializer(dim):
    '''
    Xavier weight initialization for neural networks.

    Parameters:
    - dim: Tuple (output_dim, input_dim) representing the dimensions of the weight matrix.

    Returns:
    - A numpy array of shape (output_dim, input_dim) with Xavier-initialized values.
    '''
    xavier_stddev = np.sqrt(2 / (dim[1] + dim[0]))
    return np.random.normal(0, xavier_stddev, size=(dim[0], dim[1]))

def random_initializer(dim):
    '''
    Random weight initialization for neural networks.

    Parameters:
    - dim: Tuple (output_dim, input_dim) representing the dimensions of the weight matrix.

    Returns:
    - A numpy array of shape (output_dim, input_dim) with randomly initialized values.
    '''
    return np.random.normal(0, 1, size=(dim[0], dim[1]))

def He_initializer(dim):
    '''
    He weight initialization for neural networks.

    Parameters:
    - dim: Tuple (output_dim, input_dim) representing the dimensions of the weight matrix.

    Returns:
    - A numpy array of shape (output_dim, input_dim) with He-initialized values.
    '''
    He_stddev = np.sqrt(2 / (dim[1]))
    return np.random.normal(0, 1, size=(dim[0], dim[1])) * He_stddev


def meanSquaredErrorLoss(Y_true, Y_pred):
    '''
    Calculates the Mean Squared Error (MSE) loss between true and predicted values.

    Arguments:
    - Y_true (numpy.ndarray): True output labels.
    - Y_pred (numpy.ndarray): Predicted output labels.

    Returns:
    - float: Mean Squared Error loss.
    '''
    return np.mean((Y_true - Y_pred) * (Y_true - Y_pred))

def crossEntropyLoss( Y_true, Y_pred):
    '''
    Calculates the Cross-Entropy loss between true and predicted probability distributions.

    Arguments:
    - Y_true (numpy.ndarray): True output labels in one-hot encoded form.
    - Y_pred (numpy.ndarray): Predicted probability distributions.

    Returns:
    - float: Cross-Entropy loss.
    '''
    # CE = [-Y_true[i] * np.log(Y_pred[i]) for i in range(len(Y_pred))]
    # crossEntropy = np.mean(CE)
    # return crossEntropy
    eps = 1e-15
    Y_pred = np.clip(Y_pred,eps,1.0-eps)
    loss = -np.sum(Y_true*np.log(Y_pred),axis=1)
    loss = np.mean(loss)
    return loss
# helper functions
def oneHotEncode(num_classes, Y_train_raw):
    '''
    Performs one-hot encoding on the provided labels.

    Parameters:
    - Y_train_raw (numpy.ndarray): Raw output labels.

    Returns:
    - Ydata (numpy.ndarray): One-hot encoded representation of the labels.
    '''
    res = np.zeros((num_classes, Y_train_raw.shape[0]))
    i = 0
    while(i<Y_train_raw.shape[0]):
        res[int(Y_train_raw[i])][i] = 1.0
        i+=1
    return res

def printAccuracy(epoch,trainingloss,trainingaccuracy,validationaccuracy,elapsed,alpha):
    print("Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"% (epoch,trainingloss,trainingaccuracy,validationaccuracy,elapsed,alpha,))


def accuracy(Y_true, Y_pred, data_size):
    '''
    Calculates the accuracy of the model's predictions.

    Arguments:
    - Y_true (numpy.ndarray): True output labels in one-hot encoded form.
    - Y_pred (numpy.ndarray): Predicted output labels in one-hot encoded form.
    - data_size (int): Number of samples in the dataset.

    Returns:
    - float: Accuracy of the model.
    - list: True labels.
    - list: Predicted labels.
    '''
    Y_true_vals, Y_pred_vals = [], []
    correct_vals = 0
    i = 0
    while i<data_size:
        Y_true_val = np.argmax(Y_true[:, i])
        Y_true_vals.append(Y_true_val)

        Y_pred_val = np.argmax(Y_pred[:, i])
        Y_pred_vals.append(Y_pred_val)
        if Y_true_vals[i] == Y_pred_vals[i]:
            correct_vals += 1
        i+=1
    acc = correct_vals / data_size
    return acc, Y_true_vals, Y_pred_vals


