import time
import numpy as np
from matplotlib import pyplot as plt
from my_utility import accuracy,oneHotEncode,random_initializer
from my_utility import sigmoid,tanh,reLu,del_sigmoid,del_reLu,del_tanh,softmax
from my_utility import Xavier_initializer,random_initializer,He_initializer
from my_utility import oneHotEncode,accuracy,printAccuracy
from my_utility import crossEntropyLoss,meanSquaredErrorLoss
from my_utility import SIGMOID_KEY,TANH_KEY,RELU_KEY
from my_utility import XAVIER_KEY,RANDOM_KEY,HE_KEY
from my_utility import SGD_KEY,MGD_KEY,NAG_KEY,RMSPROP_KEY,ADAM_KEY,NADAM_KEY
from my_utility import CROSS_ENTROPY_KEY,MEAN_SQUARE_KEY

GRAD_A = "del_a"
GRAD_W = "del_w"
GRAD_H = "del_h"
GRAD_B = "del_b"

class FeedForwardNeuralNetwork:
    '''
    Neural network model for feedforward architecture.

    Attributes:
    - hidden_layers (List[int]): List representing the number of neurons in each hidden layer.
    - output_layer_neuron (int): Number of neurons in the output layer.
    - X_train_raw (numpy.ndarray): Raw training input data.
    - Y_train_raw (numpy.ndarray): Raw training output labels.
    - N_train (int): Number of training samples.
    - X_val_raw (numpy.ndarray): Raw validation input data.
    - Y_val_raw (numpy.ndarray): Raw validation output labels.
    - N_val (int): Number of validation samples.
    - X_test_raw (numpy.ndarray): Raw test input data.
    - Y_test_raw (numpy.ndarray): Raw test output labels.
    - N_test (int): Number of test samples.
    - batch_size (int): Size of the mini-batch used during training.
    - weight_decay (float): Weight decay regularization parameter.
    - learning_rate (float): Learning rate for optimization.
    - epochs (int): Number of training epochs.
    - activation_fun (str): Activation function used in hidden layers.
    - initializer (str): Weight initialization method - "RANDOM" (default), "XAVIER", or "HE".
    - optimizer (str): Optimization algorithm - "SGD" (default), "MBGD", "NAGD", "RMS", "ADAM", or "NADAM".
    - loss_function (str): Loss function used for training - "CROSS_ENTROPY" (default) or MEAN_SQUARE_KEY.

    Methods:
    - __init__: Initializes the neural network with the provided parameters and initializes weights and biases.
    - initializeNeuralNet: Helper function to initialize weights and biases for the neural network layers.

    Note:
    - The network architecture is defined by the combination of hidden_layers and output_layer_neuron.
    - The input data is expected to be flattened, with dimensions (num_features, num_samples).
    - Raw input data is normalized to the range [0, 1].
    - The activation function and its derivative are specified based on the chosen activation_fun.
    - The initializer for weights is selected from "RANDOM" (default), "XAVIER", or "HE".
    - The optimization algorithm can be chosen from "SGD" (default), "MBGD", "NAGD", "RMS", "ADAM", or "NADAM".
    - The loss function for training is chosen from "CROSS_ENTROPY" (default) or MEAN_SQUARE_KEY.
    '''
    def __init__(
        self,
        num_hidden_layers,
        num_hidden_neurons,
        X_train_raw,
        Y_train_raw,
        N_train,
        X_val_raw,
        Y_val_raw,
        N_val,
        X_test_raw,
        Y_test_raw,
        N_test,
        optimizer,
        batch_size,
        weight_decay,
        learning_rate,
        max_epochs,
        activation,
        initializer,
        loss

    ):


        self.num_classes = np.max(Y_train_raw) + 1  # NUM_CLASSES

        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_neurons = num_hidden_neurons
        self.output_layer_size = self.num_classes
        self.img_height = X_train_raw.shape[1]
        self.img_width = X_train_raw.shape[2]
        self.img_flattened_size = self.img_height * self.img_width

        # self.network = layers
        self.network = (
            [self.img_flattened_size]
            + num_hidden_layers * [num_hidden_neurons]
            + [self.output_layer_size]
        )

        self.N_train = N_train
        self.N_val = N_val
        self.N_test = N_test



        self.X_train = np.transpose(
            X_train_raw.reshape(
                X_train_raw.shape[0], X_train_raw.shape[1] * X_train_raw.shape[2]
            )
        )
        self.X_test = np.transpose(
            X_test_raw.reshape(
                X_test_raw.shape[0], X_test_raw.shape[1] * X_test_raw.shape[2]
            )
        )
        self.X_val = np.transpose(
            X_val_raw.reshape(
                X_val_raw.shape[0], X_val_raw.shape[1] * X_val_raw.shape[2]
            )
        )


        self.X_train = self.X_train / 255
        self.X_test = self.X_test / 255
        self.X_val = self.X_val / 255

        self.Y_train = oneHotEncode(self.num_classes,Y_train_raw)
        self.Y_val = oneHotEncode(self.num_classes,Y_val_raw)
        self.Y_test = oneHotEncode(self.num_classes,Y_test_raw)


        self.Activations_dict = {SIGMOID_KEY: sigmoid, TANH_KEY: tanh, RELU_KEY: reLu}
        self.DerActivation_dict = {
            SIGMOID_KEY: del_sigmoid,
            TANH_KEY: del_tanh,
            RELU_KEY: del_reLu,
        }

        self.Initializer_dict = {
            XAVIER_KEY: Xavier_initializer,
            RANDOM_KEY: random_initializer,
            HE_KEY: He_initializer
        }

        self.Optimizer_dict = {
            SGD_KEY: self.sgdMiniBatch,
            MGD_KEY: self.mgd,
            NAG_KEY: self.nag,
            RMSPROP_KEY: self.rmsProp,
            ADAM_KEY: self.adam,
            NADAM_KEY: self.nadam,
        }

        self.activation = self.Activations_dict[activation]
        self.der_activation = self.DerActivation_dict[activation]
        self.optimizer = self.Optimizer_dict[optimizer]
        self.initializer = self.Initializer_dict[initializer]
        self.loss_function = loss
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.alpha = learning_rate

        self.weights, self.biases = self.initializeNeuralNet(self.network)

    def L2RegularisationLoss(self, weight_decay):
        '''
        Calculates the L2 regularization loss for the neural network weights.

        Arguments:
        - weight_decay (float): Regularization parameter.

        Returns:
        - float: L2 regularization loss.
        '''
        ALPHA = weight_decay
        return ALPHA * np.sum(
            [
                np.linalg.norm(self.weights[str(i + 1)]) ** 2
                for i in range(len(self.weights))
            ]
        )

    def predict(self,X,length_dataset):
        '''
        Generates predictions for a given input dataset.

        Arguments:
        - X (numpy.ndarray): Input dataset.
        - length_dataset (int): Number of samples in the dataset.

        Returns:
        - numpy.ndarray: Predicted output matrix.
        '''
        Y_pred = []
        i = 0
        while i < length_dataset:
            Y, H, A = self.forwardPropagate(
                X[:, i].reshape(self.img_flattened_size, 1),
                self.weights,
                self.biases,
            )
            Y_prime = Y.reshape(self.num_classes,)
            Y_pred.append(Y_prime)
            i+=1
        return np.array(Y_pred).transpose()

    def initializeNeuralNet(self, layers):
        '''
        Initializes weights and biases for the neural network layers.

        Parameters:
        - layers (List[int]): List representing the number of neurons in each layer.

        Returns:
        - weights (dict): Dictionary containing weight matrices for each layer.
        - biases (dict): Dictionary containing bias vectors for each layer.
        '''
        weights,biases = dict(),dict()
        l = 0
        while(l< len(layers) - 1):
            dummy_w = self.initializer(dim=[layers[l + 1], layers[l]])
            dummy_b = np.zeros((layers[l + 1], 1))
            key = str(l + 1)
            weights[key],biases[key] = dummy_w,dummy_b
            l+=1
        return weights, biases


    def forwardPropagate(self, X_train_batch, weights, biases):
        '''
        Performs forward propagation to calculate the output of the neural network.

        Arguments:
        - X_train_batch (numpy.ndarray): Input matrix for a batch of training data.
        - weights (dict): Dictionary containing weight matrices for each layer.
        - biases (dict): Dictionary containing bias vectors for each layer.

        Returns:
        - Y_cap (numpy.ndarray): Predicted output matrix for the given input batch.
        - H (dict): Dictionary containing activation values for each layer during forward propagation.
        - A (dict): Dictionary containing preactivation values for each layer during forward propagation.
        '''
        # Number of layers = length of weight matrix + 1
        num_layers = len(weights) + 1
        # A - Preactivations
        # H - Activations
        X = X_train_batch
        H,A = {},{}
        H["0"],A["0"] = X, X

        W = weights[str(1)]
        b = biases[str(1)]
        A[str(1)] = np.add(np.matmul(W, X), b)
        H[str(1)] = self.activation(A[str(1)])
        l = 1
        while l < num_layers - 2:
            key = str(l + 1)
            prev_layer_key = str(l)
            W,b = weights[key],biases[key]
            A[key] = np.add(np.matmul(W, H[prev_layer_key]), b)
            H[key] = self.activation(A[str(l + 1)])
            l+=1

        # Here the last layer is not activated as it is a regression problem
        last_layer_key = str(num_layers - 1)
        W,b = weights[last_layer_key],biases[last_layer_key]
        A[last_layer_key] = np.add(np.matmul(W, H[str(num_layers - 2)]), b)
        # Y = softmax(A[-1])
        Y_cap = softmax(A[last_layer_key])
        H[last_layer_key] = Y_cap
        return Y_cap, H, A

    def backPropagate(
        self, Y, H, A, Y_train_batch, weight_decay=0
    ):
        '''
        Backpropagate through the neural network to compute gradients with respect to weights and biases.

        Parameters:
        - y_cap: The predicted output of the neural network.
        - H: Dictionary containing hidden layer outputs.
        - A: Dictionary containing pre-activation values for each layer.
        - y_true: The true output labels.
        - weight_decay: Regularization parameter to control overfitting (default is 0).

        Returns:
        - del_w: List of weight gradients for each layer.
        - del_b: List of bias gradients for each layer.
        '''

        ALPHA = weight_decay
        gradients_weights = []
        gradients_biases = []
        num_layers = len(self.network)

        # Gradient with respect to the output layer is absolutely fine.
        if self.loss_function == CROSS_ENTROPY_KEY:
            globals()["grad_a" + str(num_layers - 1)] = -(Y_train_batch - Y)
        elif self.loss_function == MEAN_SQUARE_KEY:
            globals()["grad_a" + str(num_layers - 1)] = np.multiply(
                2 * (Y - Y_train_batch), np.multiply(Y, (1 - Y))
            )

        for l in range(num_layers - 2, -1, -1):

            if ALPHA != 0:
                globals()["grad_W" + str(l + 1)] = (
                    np.outer(globals()["grad_a" + str(l + 1)], H[str(l)])
                    + ALPHA * self.weights[str(l + 1)]
                )
            elif ALPHA == 0:
                globals()["grad_W" + str(l + 1)] = np.outer(
                    globals()["grad_a" + str(l + 1)], H[str(l)]
                )
            globals()["grad_b" + str(l + 1)] = globals()["grad_a" + str(l + 1)]
            gradients_weights.append(globals()["grad_W" + str(l + 1)])
            gradients_biases.append(globals()["grad_b" + str(l + 1)])
            if l != 0:
                globals()["grad_h" + str(l)] = np.matmul(
                    self.weights[str(l + 1)].transpose(),
                    globals()["grad_a" + str(l + 1)],
                )
                globals()["grad_a" + str(l)] = np.multiply(
                    globals()["grad_h" + str(l)], self.der_activation(A[str(l)])
                )
            elif l == 0:

                globals()["grad_h" + str(l)] = np.matmul(
                    self.weights[str(l + 1)].transpose(),
                    globals()["grad_a" + str(l + 1)],
                )
                globals()["grad_a" + str(l)] = np.multiply(
                    globals()["grad_h" + str(l)], (A[str(l)])
                )
        return gradients_weights, gradients_biases

    #Optimisers defined here onwards
    def sgd(self, epochs, length_dataset, learning_rate, weight_decay=0):
        '''
        Implement Stochastic Gradient Descent (SGD) optimization for training the neural network.

        Parameters:
        - epochs: Number of training epochs.
        - length_dataset: Number of samples in the training dataset.
        - learning_rate: Learning rate for the optimization.
        - weight_decay: Regularization parameter to control overfitting (default is 0).

        Returns:
        - trainingloss: List of training losses per epoch.
        - trainingaccuracy: List of training accuracies per epoch.
        - validationaccuracy: List of validation accuracies per epoch.
        - Y_pred: Predicted outputs after training.
        '''
        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        network_size = len(self.network)

        # Extract a subset of the training dataset
        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]
        epoch = 0

        while epoch < epochs:
            start_time = time.time()

            # Reshape input and target arrays
            X_train, Y_train = X_train.reshape(
                self.img_flattened_size, length_dataset
            ), Y_train.reshape(self.num_classes, length_dataset)

            LOSS = []
            del_w = [np.zeros((self.network[l + 1], self.network[l])) for l in range(0, len(self.network) - 1)]
            del_b = [np.zeros((self.network[l + 1], 1)) for l in range(0, len(self.network) - 1)]

            i = 0

            # Iterate through the dataset
            while i < length_dataset:
                Y_cap, H, A = self.forwardPropagate(
                    X_train[:, i].reshape(self.img_flattened_size, 1), self.weights, self.biases
                )
                grad_weights, grad_biases = self.backPropagate(
                    Y_cap, H, A, Y_train[:, i].reshape(self.num_classes, 1)
                )

                del_w = [grad_weights[network_size - 2 - i] for i in range(network_size - 1)]
                del_b = [grad_biases[network_size - 2 - i] for i in range(network_size - 1)]

                l2Loss = self.L2RegularisationLoss(weight_decay)
                if self.loss_function == MEAN_SQUARE_KEY:
                    LOSS.append(meanSquaredErrorLoss(self.Y_train[:, i].reshape(self.num_classes, 1), Y_cap) + l2Loss)
                else:
                    LOSS.append(crossEntropyLoss(self.Y_train[:, i].reshape(self.num_classes, 1), Y_cap) + l2Loss)

                # Update weights and biases
                for j in range(len(self.weights)):
                    self.weights[str(j + 1)] = self.weights[str(j + 1)] - learning_rate * del_w[j]

                for j in range(len(self.biases)):
                    self.biases[str(j + 1)] = self.biases[str(j + 1)] - learning_rate * del_b[j]

                i += 1

            elapsed = time.time() - start_time

            # Compute training and validation accuracies, and append to lists
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            Y_val_pred = self.predict(self.X_val, self.N_val)
            validationaccuracy.append(accuracy(self.Y_val, Y_val_pred, self.N_val)[0])
            l2RegulerizedValue = self.L2RegularisationLoss(weight_decay)
            val_loss = 0
            if self.loss_function == MEAN_SQUARE_KEY:
              temp = meanSquaredErrorLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            else:
              temp = crossEntropyLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            printAccuracy(epoch,trainingloss[epoch],trainingaccuracy[epoch],validationaccuracy[epoch],elapsed,self.alpha)
            epoch += 1

        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred




    def sgdMiniBatch(self, epochs,length_dataset, batch_size, learning_rate, weight_decay = 0):
        '''
        Train the neural network using Stochastic Gradient Descent (SGD) with Mini-Batch updates.

        Parameters:
        - epochs (int): Number of training epochs.
        - length_dataset (int): Number of samples in the training dataset.
        - batch_size (int): Size of each mini-batch during training.
        - learning_rate (float): The learning rate for updating weights and biases.
        - weight_decay (float, optional): L2 regularization term to control overfitting (default is 0).

        Returns:
        - trainingloss (list): List of training losses for each epoch.
        - trainingaccuracy (list): List of training accuracies for each epoch.
        - validationaccuracy (list): List of validation accuracies for each epoch.
        - Y_pred (numpy array): Predicted labels for the validation set after training.

        '''
        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]

        trainingloss = []
        trainingaccuracy, validationaccuracy = [], []

        num_points_seen = 0

        num_layers = len(self.network)
        epoch = 0
        while epoch < epochs:
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)

            LOSS = []
            #Y_pred = []
            deltaw,deltab = [],[]
            for l in range(0, len(self.network)-1):
                deltaw.append(np.zeros((self.network[l+1], self.network[l])))
            for l in range(0, len(self.network)-1):
                deltab.append(np.zeros((self.network[l+1], 1)))

            i = 0
            while i < length_dataset:
                Y,H,A = self.forwardPropagate(X_train[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases)
                grad_weights, grad_biases = self.backPropagate(Y,H,A,Y_train[:,i].reshape(self.num_classes,1))

                tempDeltaw,tempDeltab = [],[]
                for j in range(num_layers - 1):
                    tempDeltaw.append(grad_weights[num_layers-2 - j] + deltaw[j])

                for j in range(num_layers - 1):
                    tempDeltab.append(grad_biases[num_layers-2 - j] + deltab[j])
                deltaw,deltab = tempDeltaw,tempDeltab
                l2RegulerizedValue = self.L2RegularisationLoss(weight_decay)
                if self.loss_function == MEAN_SQUARE_KEY:
                    LOSS.append(meanSquaredErrorLoss(
                            self.Y_train[:, i].reshape(self.num_classes, 1), Y
                        )
                        + l2RegulerizedValue
                        )
                else:
                    LOSS.append(
                        crossEntropyLoss(
                            self.Y_train[:, i].reshape(self.num_classes, 1), Y
                        )
                        + l2RegulerizedValue
                    )
                num_points_seen +=1

                if int(num_points_seen) % batch_size == 0:


                    self.weights = {str(i+1):(self.weights[str(i+1)] - learning_rate*deltaw[i]/batch_size) for i in range(len(self.weights))}
                    self.biases = {str(i+1):(self.biases[str(i+1)] - learning_rate*deltab[i]) for i in range(len(self.biases))}

                    #resetting gradient updates
                    deltaw = []
                    for l in range(0, len(self.network)-1):
                        deltaw.append(np.zeros((self.network[l+1], self.network[l])))
                    deltab = []
                    for l in range(0, len(self.network)-1):
                        deltab.append(np.zeros((self.network[l+1], 1)))
                i+=1
            elapsed = time.time() - start_time

            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            Y_val_pred = self.predict(self.X_val, self.N_val)
            validationaccuracy.append(accuracy(self.Y_val, Y_val_pred, self.N_val)[0])
            l2RegulerizedValue = self.L2RegularisationLoss(weight_decay)
            val_loss = 0
            if self.loss_function == MEAN_SQUARE_KEY:
              temp = meanSquaredErrorLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            else:
              temp = crossEntropyLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            printAccuracy(epoch,trainingloss[epoch],trainingaccuracy[epoch],validationaccuracy[epoch],elapsed,self.alpha)
            epoch+=1


        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred

    def mgd(self, epochs,length_dataset, batch_size, learning_rate, weight_decay = 0):
        '''
        Train the neural network using the Mini-Batch Gradient Descent (MGD) optimization algorithm with momentum.

        Parameters:
        - epochs (int): Number of training epochs.
        - length_dataset (int): Number of samples in the training dataset.
        - batch_size (int): Size of each mini-batch during training.
        - learning_rate (float): The learning rate for updating weights and biases.
        - weight_decay (float, optional): L2 regularization term to control overfitting (default is 0).

        Returns:
        - trainingloss (list): List of training losses for each epoch.
        - trainingaccuracy (list): List of training accuracies for each epoch.
        - validationaccuracy (list): List of validation accuracies for each epoch.
        - Y_pred (numpy array): Predicted labels for the validation set after training.

        '''

        GAMMA = 0.9

        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]


        trainingloss = []
        trainingaccuracy, validationaccuracy = [], []

        num_layers = len(self.network)

        prev_v_w,prev_v_b = [],[]
        for l in range(0, len(self.network)-1):
            prev_v_w.append(np.zeros((self.network[l+1], self.network[l])))
        for l in range(0, len(self.network)-1):
            prev_v_b.append(np.zeros((self.network[l+1], 1)))

        num_points_seen = 0
        epoch = 0
        while epoch < epochs:
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)

            LOSS = []

            deltaw = []
            for l in range(0, len(self.network)-1):
                deltaw.append(np.zeros((self.network[l+1], self.network[l])))

            deltab = []
            for l in range(0, len(self.network)-1):
                deltab.append(np.zeros((self.network[l+1], 1)))

            for i in range(length_dataset):
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases)
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(self.num_classes,1))

                tempDeltaw,tempDeltab = [],[]
                for j in range(num_layers - 1):
                    tempDeltaw.append(grad_weights[num_layers-2 - j] + deltaw[j])

                for j in range(num_layers - 1):
                    tempDeltab.append(grad_biases[num_layers-2 - j] + deltab[j])
                deltaw,deltab = tempDeltaw,tempDeltab

                l2RegulerizedValue = self.L2RegularisationLoss(weight_decay)
                if self.loss_function == MEAN_SQUARE_KEY:
                    LOSS.append(meanSquaredErrorLoss(
                            self.Y_train[:, i].reshape(self.num_classes, 1), Y
                        )
                        + l2RegulerizedValue
                        )
                else:
                    LOSS.append(
                        crossEntropyLoss(
                            self.Y_train[:, i].reshape(self.num_classes, 1), Y
                        )
                        + l2RegulerizedValue
                    )
                num_points_seen +=1

                if int(num_points_seen) % batch_size == 0:

                    v_w = [GAMMA*prev_v_w[i] + learning_rate*deltaw[i]/batch_size for i in range(num_layers - 1)]
                    v_b = [GAMMA*prev_v_b[i] + learning_rate*deltab[i]/batch_size for i in range(num_layers - 1)]
                    tempWeights = {}
                    tempBiases = {}
                    for l in range(len(self.weights)):
                        tempWeights[str(l+1)] = self.weights[str(l+1)] - v_w[l]
                    self.weights = tempWeights

                    for l in range(len(self.biases)):
                        tempBiases[str(l+1)] = self.biases[str(l+1)] - v_b[l]
                    self.biases = tempBiases

                    prev_v_w = v_w
                    prev_v_b = v_b

                    deltaw = []
                    for l in range(0, len(self.network)-1):
                        deltaw.append(np.zeros((self.network[l+1], self.network[l])))
                    deltab = []
                    for l in range(0, len(self.network)-1):
                        deltab.append(np.zeros((self.network[l+1], 1)))

            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            Y_val_pred = self.predict(self.X_val, self.N_val)
            validationaccuracy.append(accuracy(self.Y_val, Y_val_pred, self.N_val)[0])
            l2RegulerizedValue = self.L2RegularisationLoss(weight_decay)
            val_loss = 0
            if self.loss_function == MEAN_SQUARE_KEY:
              temp = meanSquaredErrorLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            else:
              temp = crossEntropyLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            printAccuracy(epoch,trainingloss[epoch],trainingaccuracy[epoch],validationaccuracy[epoch],elapsed,self.alpha)
            epoch+=1

        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred



    def nag(self,epochs,length_dataset, batch_size,learning_rate, weight_decay = 0):
        '''
        Train the neural network using the Nesterov Accelerated Gradient (NAG) optimization algorithm.

        Parameters:
        - epochs (int): Number of training epochs.
        - length_dataset (int): Number of samples in the training dataset.
        - batch_size (int): Size of each mini-batch during training.
        - learning_rate (float): The learning rate for updating weights and biases.
        - weight_decay (float, optional): L2 regularization term to control overfitting (default is 0).

        Returns:
        - trainingloss (list): List of training losses for each epoch.
        - trainingaccuracy (list): List of training accuracies for each epoch.
        - validationaccuracy (list): List of validation accuracies for each epoch.
        - Y_pred (numpy array): Predicted labels for the validation set after training.

        '''
        GAMMA = 0.9

        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]


        trainingloss = []
        trainingaccuracy, validationaccuracy = [], []

        num_layers = len(self.network)

        prev_v_w,prev_v_b = [],[]
        for l in range(0, len(self.network)-1):
            prev_v_w.append(np.zeros((self.network[l+1], self.network[l])))
        for l in range(0, len(self.network)-1):
            prev_v_b.append(np.zeros((self.network[l+1], 1)))

        num_points_seen = 0
        epoch = 0
        while epoch < epochs:
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)

            LOSS = []
            #Y_pred = []

            deltaw = []
            for l in range(0, len(self.network)-1):
                deltaw.append(np.zeros((self.network[l+1], self.network[l])))

            deltab = []
            for l in range(0, len(self.network)-1):
                deltab.append(np.zeros((self.network[l+1], 1)))

            v_w = [GAMMA*prev_v_w[l] for l in range(0, len(self.network)-1)]
            v_b = [GAMMA*prev_v_b[l] for l in range(0, len(self.network)-1)]
            i = 0
            while i < length_dataset:
                winter = dict()
                for l in range(0, len(self.network)-1):
                    winter[str(l+1)] = self.weights[str(l+1)] - v_w[l]

                binter = dict()
                for l in range(0, len(self.network)-1):
                    binter[str(l+1)] = self.biases[str(l+1)] - v_b[l]

                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(self.img_flattened_size,1), winter, binter)
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(self.num_classes,1))

                tempDeltaw,tempDeltab = [],[]
                for l in range(num_layers - 1):
                    tempDeltaw.append(grad_weights[num_layers-2 - l] + deltaw[l])

                for l in range(num_layers - 1):
                    tempDeltab.append(grad_biases[num_layers-2 - l] + deltab[l])
                deltaw,deltab = tempDeltaw,tempDeltab

                l2RegulerizedValue = self.L2RegularisationLoss(weight_decay)
                if self.loss_function == MEAN_SQUARE_KEY:
                    LOSS.append(meanSquaredErrorLoss(
                            self.Y_train[:, i].reshape(self.num_classes, 1), Y
                        )
                        + l2RegulerizedValue
                        )
                else:
                    LOSS.append(
                        crossEntropyLoss(
                            self.Y_train[:, i].reshape(self.num_classes, 1), Y
                        )
                        + l2RegulerizedValue
                    )

                num_points_seen +=1

                if int(num_points_seen) % batch_size == 0:

                    v_w = [GAMMA*prev_v_w[i] + learning_rate*deltaw[i]/batch_size for i in range(num_layers - 1)]
                    v_b = [GAMMA*prev_v_b[i] + learning_rate*deltab[i]/batch_size for i in range(num_layers - 1)]

                    tempW = dict()
                    for l in range(len(self.weights)):
                        tempW[str(l+1)] = self.weights[str(l+1)] - v_w[l]

                    tempB = dict()
                    for l in range(len(self.biases)):
                        tempB[str(l+1)] = self.biases[str(l+1)] - v_b[l]

                    self.weights,self.biases = tempW,tempB
                    prev_v_w, prev_v_b = v_w, v_b

                    deltaw = []
                    for l in range(0, len(self.network)-1):
                        deltaw.append(np.zeros((self.network[l+1], self.network[l])))
                    deltab = []
                    for l in range(0, len(self.network)-1):
                        deltab.append(np.zeros((self.network[l+1], 1)))

                i+=1


            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            Y_val_pred = self.predict(self.X_val, self.N_val)
            validationaccuracy.append(accuracy(self.Y_val, Y_val_pred, self.N_val)[0])
            l2RegulerizedValue = self.L2RegularisationLoss(weight_decay)
            val_loss = 0
            if self.loss_function == MEAN_SQUARE_KEY:
              temp = meanSquaredErrorLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            else:
              temp = crossEntropyLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            printAccuracy(epoch,trainingloss[epoch],trainingaccuracy[epoch],validationaccuracy[epoch],elapsed,self.alpha)
            epoch+=1
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred


    def rmsProp(self, epochs,length_dataset, batch_size, learning_rate, weight_decay = 0):
        '''
        Train the neural network using the RMSProp optimization algorithm.

        Parameters:
        - epochs (int): Number of training epochs.
        - length_dataset (int): Number of samples in the training dataset.
        - batch_size (int): Size of each mini-batch during training.
        - learning_rate (float): The learning rate for updating weights and biases.
        - weight_decay (float, optional): L2 regularization term to control overfitting (default is 0).

        Returns:
        - trainingloss (list): List of training losses for each epoch.
        - trainingaccuracy (list): List of training accuracies for each epoch.
        - validationaccuracy (list): List of validation accuracies for each epoch.
        - Y_pred (numpy array): Predicted labels for the validation set after training.

        '''

        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]


        trainingloss = []
        trainingloss = []
        trainingaccuracy, validationaccuracy = [], []

        num_layers = len(self.network)
        EPS, BETA = 1e-8, 0.9

        v_w,v_b = [],[]
        for l in range(0, len(self.network)-1):
            v_w.append(np.zeros((self.network[l+1], self.network[l])))
        for l in range(0, len(self.network)-1):
            v_b.append(np.zeros((self.network[l+1], 1)))

        num_points_seen = 0
        epoch = 0
        while epoch < epochs:
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)


            LOSS = []
            #Y_pred = []

            deltaw = []
            for l in range(0, len(self.network)-1):
                deltaw.append(np.zeros((self.network[l+1], self.network[l])))

            deltab = []
            for l in range(0, len(self.network)-1):
                deltab.append(np.zeros((self.network[l+1], 1)))
            i = 0
            while i < length_dataset:

                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases)
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(self.num_classes,1))

                tempDeltaw,tempDeltab = [],[]
                for j in range(num_layers - 1):
                    tempDeltaw.append(grad_weights[num_layers-2 - j] + deltaw[j])

                for j in range(num_layers - 1):
                    tempDeltab.append(grad_biases[num_layers-2 - j] + deltab[j])
                deltaw,deltab = tempDeltaw,tempDeltab

                l2RegulerizedValue = self.L2RegularisationLoss(weight_decay)
                if self.loss_function == MEAN_SQUARE_KEY:
                    LOSS.append(meanSquaredErrorLoss(
                            self.Y_train[:, i].reshape(self.num_classes, 1), Y
                        )
                        + l2RegulerizedValue
                        )
                else:
                    LOSS.append(
                        crossEntropyLoss(
                            self.Y_train[:, i].reshape(self.num_classes, 1), Y
                        )
                        + l2RegulerizedValue
                    )

                num_points_seen +=1

                if int(num_points_seen) % batch_size == 0:

                    v_w = [BETA*v_w[i] + (1-BETA)*(deltaw[i])**2 for i in range(num_layers - 1)]
                    v_b = [BETA*v_b[i] + (1-BETA)*(deltab[i])**2 for i in range(num_layers - 1)]

                    tempW = dict()
                    for l in range(len(self.weights)):
                        tempW[str(l+1)] = self.weights[str(l+1)] - deltaw[l]*(learning_rate/np.sqrt(v_w[l]+EPS))

                    tempB = dict()
                    for l in range(len(self.biases)):
                        tempB[str(l+1)] = self.biases[str(l+1)] - deltab[l]*(learning_rate/np.sqrt(v_b[l]+EPS))


                    self.weights,self.biases = tempW,tempB

                    deltaw = []
                    for l in range(0, len(self.network)-1):
                        deltaw.append(np.zeros((self.network[l+1], self.network[l])))
                    deltab = []
                    for l in range(0, len(self.network)-1):
                        deltab.append(np.zeros((self.network[l+1], 1)))
                i+=1

            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            Y_val_pred = self.predict(self.X_val, self.N_val)
            validationaccuracy.append(accuracy(self.Y_val, Y_val_pred, self.N_val)[0])
            l2RegulerizedValue = self.L2RegularisationLoss(weight_decay)
            val_loss = 0
            if self.loss_function == MEAN_SQUARE_KEY:
              temp = meanSquaredErrorLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            else:
              temp = crossEntropyLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            printAccuracy(epoch,trainingloss[epoch],trainingaccuracy[epoch],validationaccuracy[epoch],elapsed,self.alpha)
            epoch+=1

        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred

    def adam(self, epochs,length_dataset, batch_size, learning_rate, weight_decay = 0):
        '''
        Train the neural network using the Adam optimization algorithm.

        Parameters:
        - epochs (int): Number of training epochs.
        - length_dataset (int): Number of samples in the training dataset.
        - batch_size (int): Size of each mini-batch during training.
        - learning_rate (float): The learning rate for updating weights and biases.
        - weight_decay (float, optional): L2 regularization term to control overfitting (default is 0).

        Returns:
        - trainingloss (list): List of training losses for each epoch.
        - trainingaccuracy (list): List of training accuracies for each epoch.
        - validationaccuracy (list): List of validation accuracies for each epoch.
        - Y_pred (numpy array): Predicted labels for the validation set after training.

        '''
        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]

        trainingloss = []
        trainingaccuracy, validationaccuracy = [], []
        num_layers = len(self.network)
        EPS, BETA1, BETA2 = 1e-8, 0.9, 0.99


        m_w,m_b = [],[]
        for l in range(0, len(self.network)-1):
            m_w.append(np.zeros((self.network[l+1], self.network[l])))
        for l in range(0, len(self.network)-1):
            m_b.append(np.zeros((self.network[l+1], 1)))

        v_w,v_b = [],[]
        for l in range(0, len(self.network)-1):
            v_w.append(np.zeros((self.network[l+1], self.network[l])))
        for l in range(0, len(self.network)-1):
            v_b.append(np.zeros((self.network[l+1], 1)))

        m_w_hat,m_b_hat = [],[]
        for l in range(0, len(self.network)-1):
            m_w_hat.append(np.zeros((self.network[l+1], self.network[l])))
        for l in range(0, len(self.network)-1):
            m_b_hat.append(np.zeros((self.network[l+1], 1)))

        v_w_hat,v_b_hat = [],[]
        for l in range(0, len(self.network)-1):
            v_w_hat.append(np.zeros((self.network[l+1], self.network[l])))
        for l in range(0, len(self.network)-1):
            v_b_hat.append(np.zeros((self.network[l+1], 1)))

        num_points_seen = 0
        epoch = 0
        while epoch < epochs:
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)


            LOSS = []
            #Y_pred = []

            deltaw = []
            for l in range(0, len(self.network)-1):
                deltaw.append(np.zeros((self.network[l+1], self.network[l])))

            deltab = []
            for l in range(0, len(self.network)-1):
                deltab.append(np.zeros((self.network[l+1], 1)))
            i = 0
            while i < length_dataset:
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases)
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(self.num_classes,1))

                tempDeltaw,tempDeltab = [],[]
                for j in range(num_layers - 1):
                    tempDeltaw.append(grad_weights[num_layers-2 - j] + deltaw[j])

                for j in range(num_layers - 1):
                    tempDeltab.append(grad_biases[num_layers-2 - j] + deltab[j])
                deltaw,deltab = tempDeltaw,tempDeltab

                l2RegulerizedValue = self.L2RegularisationLoss(weight_decay)
                if self.loss_function == MEAN_SQUARE_KEY:
                    LOSS.append(meanSquaredErrorLoss(
                            self.Y_train[:, i].reshape(self.num_classes, 1), Y
                        )
                        + l2RegulerizedValue
                        )
                else:
                    LOSS.append(
                        crossEntropyLoss(
                            self.Y_train[:, i].reshape(self.num_classes, 1), Y
                        )
                        + l2RegulerizedValue
                    )

                num_points_seen += 1
                ctr = 0
                if int(num_points_seen) % batch_size == 0:
                    ctr += 1

                    tempW,tempB = [],[]
                    for l in range(0, len(self.network)-1):
                        tempW.append(BETA1*m_w[l] + (1-BETA1)*deltaw[l])
                    for l in range(0, len(self.network)-1):
                        tempB.append(BETA1*m_b[l] + (1-BETA1)*deltab[l])
                    m_w,m_b = tempW,tempB

                    tempW,tempB = [],[]
                    for l in range(0, len(self.network)-1):
                        tempW.append(BETA2*v_w[l] + (1-BETA2)*(deltaw[l])**2)
                    for l in range(0, len(self.network)-1):
                        tempB.append(BETA2*v_b[l] + (1-BETA2)*(deltab[l])**2)
                    v_w,v_b = tempW,tempB

                    tempW,tempB = [],[]
                    for l in range(0, len(self.network)-1):
                        tempW.append(m_w[l]/(1-BETA1**(epoch+1)))
                    for l in range(0, len(self.network)-1):
                        tempB.append(m_b[l]/(1-BETA1**(epoch+1)))
                    m_w_hat,m_b_hat = tempW,tempB


                    tempW,tempB = [],[]
                    for l in range(0, len(self.network)-1):
                        tempW.append(v_w[l]/(1-BETA2**(epoch+1)))
                    for l in range(0, len(self.network)-1):
                        tempB.append(v_b[l]/(1-BETA2**(epoch+1)))
                    v_w_hat,v_b_hat = tempW,tempB

                    tempW = dict()
                    for l in range(len(self.weights)):
                        tempW[str(l+1)] = self.weights[str(l+1)] - (learning_rate/np.sqrt(v_w[l]+EPS))*m_w_hat[l]
                    tempB = dict()
                    for l in range(len(self.biases)):
                        tempB[str(l+1)] = self.biases[str(l+1)] - (learning_rate/np.sqrt(v_b[l]+EPS))*m_b_hat[l]
                    self.weights,self.biases = tempW,tempB
                    deltaw = []
                    for l in range(0, len(self.network)-1):
                        deltaw.append(np.zeros((self.network[l+1], self.network[l])))
                    deltab = []
                    for l in range(0, len(self.network)-1):
                        deltab.append(np.zeros((self.network[l+1], 1)))
                i+=1

            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            Y_val_pred = self.predict(self.X_val, self.N_val)
            validationaccuracy.append(accuracy(self.Y_val, Y_val_pred, self.N_val)[0])
            l2RegulerizedValue = self.L2RegularisationLoss(weight_decay)
            val_loss = 0
            if self.loss_function == MEAN_SQUARE_KEY:
              temp = meanSquaredErrorLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            else:
              temp = crossEntropyLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            printAccuracy(epoch,trainingloss[epoch],trainingaccuracy[epoch],validationaccuracy[epoch],elapsed,self.alpha)
            epoch+=1

        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred


    def nadam(self, epochs,length_dataset, batch_size, learning_rate, weight_decay = 0):
        '''
        Train the neural network using the Nadam optimization algorithm.

        Parameters:
        - epochs (int): Number of training epochs.
        - length_dataset (int): Number of samples in the training dataset.
        - batch_size (int): Size of each mini-batch during training.
        - learning_rate (float): The learning rate for updating weights and biases.
        - weight_decay (float, optional): L2 regularization term to control overfitting (default is 0).

        Returns:
        - trainingloss (list): List of training losses for each epoch.
        - trainingaccuracy (list): List of training accuracies for each epoch.
        - validationaccuracy (list): List of validation accuracies for each epoch.
        - Y_pred (numpy array): Predicted labels for the validation set after training.

        '''
        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]


        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        num_layers = len(self.network)

        GAMMA, EPS, BETA1, BETA2 = 0.9, 1e-8, 0.9, 0.99

        m_w,m_b = [],[]
        for l in range(0, len(self.network)-1):
            m_w.append(np.zeros((self.network[l+1], self.network[l])))
        for l in range(0, len(self.network)-1):
            m_b.append(np.zeros((self.network[l+1], 1)))

        v_w,v_b = [],[]
        for l in range(0, len(self.network)-1):
            v_w.append(np.zeros((self.network[l+1], self.network[l])))
        for l in range(0, len(self.network)-1):
            v_b.append(np.zeros((self.network[l+1], 1)))

        m_w_hat,m_b_hat = [],[]
        for l in range(0, len(self.network)-1):
            m_w_hat.append(np.zeros((self.network[l+1], self.network[l])))
        for l in range(0, len(self.network)-1):
            m_b_hat.append(np.zeros((self.network[l+1], 1)))

        v_w_hat,v_b_hat = [],[]
        for l in range(0, len(self.network)-1):
            v_w_hat.append(np.zeros((self.network[l+1], self.network[l])))
        for l in range(0, len(self.network)-1):
            v_b_hat.append(np.zeros((self.network[l+1], 1)))

        num_points_seen = 0

        epoch = 0
        while epoch < epochs:
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)

            LOSS = []
            #Y_pred = []
            deltaw = []
            for l in range(0, len(self.network)-1):
                deltaw.append(np.zeros((self.network[l+1], self.network[l])))

            deltab = []
            for l in range(0, len(self.network)-1):
                deltab.append(np.zeros((self.network[l+1], 1)))
            i = 0
            while i < length_dataset:

                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases)
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(self.num_classes,1))

                tempDeltaw,tempDeltab = [],[]
                for j in range(num_layers - 1):
                    tempDeltaw.append(grad_weights[num_layers-2 - j] + deltaw[j])

                for j in range(num_layers - 1):
                    tempDeltab.append(grad_biases[num_layers-2 - j] + deltab[j])
                deltaw,deltab = tempDeltaw,tempDeltab


                l2RegulerizedValue = self.L2RegularisationLoss(weight_decay)
                if self.loss_function == MEAN_SQUARE_KEY:
                    LOSS.append(meanSquaredErrorLoss(
                            self.Y_train[:, i].reshape(self.num_classes, 1), Y
                        )
                        + l2RegulerizedValue
                        )
                else:
                    LOSS.append(
                        crossEntropyLoss(
                            self.Y_train[:, i].reshape(self.num_classes, 1), Y
                        )
                        + l2RegulerizedValue
                    )

                num_points_seen += 1

                if num_points_seen % batch_size == 0:

                    tempW,tempB = [],[]
                    for l in range(0, len(self.network)-1):
                        tempW.append(BETA1*m_w[l] + (1-BETA1)*deltaw[l])
                    for l in range(0, len(self.network)-1):
                        tempB.append(BETA1*m_b[l] + (1-BETA1)*deltab[l])
                    m_w,m_b = tempW,tempB

                    tempW,tempB = [],[]
                    for l in range(0, len(self.network)-1):
                        tempW.append(BETA2*v_w[l] + (1-BETA2)*(deltaw[l])**2)
                    for l in range(0, len(self.network)-1):
                        tempB.append(BETA2*v_b[l] + (1-BETA2)*(deltab[l])**2)
                    v_w,v_b = tempW,tempB

                    tempW,tempB = [],[]
                    for l in range(0, len(self.network)-1):
                        tempW.append(m_w[l]/(1-BETA1**(epoch+1)))
                    for l in range(0, len(self.network)-1):
                        tempB.append(m_b[l]/(1-BETA1**(epoch+1)))
                    m_w_hat,m_b_hat = tempW,tempB


                    tempW,tempB = [],[]
                    for l in range(0, len(self.network)-1):
                        tempW.append(v_w[l]/(1-BETA2**(epoch+1)))
                    for l in range(0, len(self.network)-1):
                        tempB.append(v_b[l]/(1-BETA2**(epoch+1)))
                    v_w_hat,v_b_hat = tempW,tempB

                    tempW = dict()
                    for l in range(len(self.weights)):
                        tempW[str(l+1)] = self.weights[str(l+1)] - (learning_rate/(np.sqrt(v_w_hat[l])+EPS))*(BETA1*m_w_hat[l]+ (1-BETA1)*deltaw[l])
                    tempB = dict()
                    for l in range(len(self.biases)):
                        tempB[str(l+1)] = self.biases[str(l+1)] - (learning_rate/np.sqrt(v_b_hat[l])+EPS)*(BETA1*m_b_hat[l] + (1-BETA1)*deltab[l])
                    self.weights,self.biases = tempW,tempB

                    deltaw = []
                    for l in range(0, len(self.network)-1):
                        deltaw.append(np.zeros((self.network[l+1], self.network[l])))
                    deltab = []
                    for l in range(0, len(self.network)-1):
                        deltab.append(np.zeros((self.network[l+1], 1)))
                i+=1

            elapsed = time.time() - start_time

            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(LOSS))
            trainingaccuracy.append(accuracy(Y_train, Y_pred, length_dataset)[0])
            Y_val_pred = self.predict(self.X_val, self.N_val)
            validationaccuracy.append(accuracy(self.Y_val, Y_val_pred, self.N_val)[0])
            l2RegulerizedValue = self.L2RegularisationLoss(weight_decay)
            val_loss = 0
            if self.loss_function == MEAN_SQUARE_KEY:
              temp = meanSquaredErrorLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            else:
              temp = crossEntropyLoss(self.Y_val.T, Y_val_pred.T)+ l2RegulerizedValue
              val_loss = np.mean(temp)
            printAccuracy(epoch,trainingloss[epoch],trainingaccuracy[epoch],validationaccuracy[epoch],elapsed,self.alpha)
            epoch+=1
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred
