import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import argparse
from feedForwardNeuralNetwork import FeedForwardNeuralNetwork

parser = argparse.ArgumentParser()
parser.add_argument("-wp","--wandb_project",help="Project name used to track experiments in Weights & Biases dashboard",default="dl-assigment-1")
parser.add_argument("-we","--wandb_entity",help="Wandb Entity used to track experiments in the Weights & Biases dashboard.",default="cs23m007")
parser.add_argument("-d","--dataset",help="choices: ['mnist', 'fashion_mnist']",choices=['mnist', 'fashion_mnist'],default="fashion_mnist")
parser.add_argument("-e","--epochs",help="Number of epochs to train neural network.",choices=[5,10],default=10)
parser.add_argument("-b","--batch_size",help="Batch size used to train neural network.",choices=[16,32,64],default=32)
parser.add_argument("-l","--loss",help="choices: ['mean_squared_error', 'cross_entropy']",choices=['cross_entropy','mean_squared_error'],default='cross_entropy')
parser.add_argument("-o","--optimizer",help="choices: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']",choices=['sgd','momentum','nag','rmsprop','adam','nadam'],default='nadam')
parser.add_argument("-lr","--learning_rate",help="Learning rate used to optimize model parameters",choices=['1e-3','1e-4'],default='1e-3')
parser.add_argument("-m","--momentum",help="Momentum used by momentum and nag optimizers.",choices=['0.5'],default=0.5)
parser.add_argument("-beta","--beta",help="Beta used by rmsprop optimizer",choices=['0.5'],default=0.5)
parser.add_argument("-beta1","--beta1",help="Beta1 used by adam and nadam optimizers.",choices=['0.5'],default=0.5)
parser.add_argument("-beta2","--beta2",help="Beta2 used by adam and nadam optimizers.",choices=['0.5'],default=0.5)
parser.add_argument("-eps","--epsilon",help="Epsilon used by optimizers.",choices=['0.000001'],default=0.000001)
parser.add_argument("-w_d","--weight_decay",help="Weight decay used by optimizers.",choices=['0','0.0005','0.5'],default=0)
parser.add_argument("-w_i","--weight_init",help="choices: ['random', 'Xavier']",choices=['random','Xavier'],default='Xavier')
parser.add_argument("-nhl","--num_layers",help="Number of hidden layers used in feedforward neural network.",choices=['3','4','5'],default=3)
parser.add_argument("-sz","--hidden_size",help="Number of hidden neurons in a feedforward layer.",choices=['32','64','128'],default=128)
parser.add_argument("-a","--activation",help="choices: ['identity', 'sigmoid', 'tanh', 'ReLU']",choices=['sigmoid','tanh','ReLU'],default='ReLu')
args = parser.parse_args()

(trainIn, trainOut), (testIn, testOut) = fashion_mnist.load_data()
if(args.dataset=='mnist'):
    (trainIn, trainOut), (testIn, testOut) = mnist.load_data()

N_train_full = trainOut.shape[0]
N_train = int(0.9*N_train_full)
N_validation = int(0.1 * trainOut.shape[0])
N_test = testOut.shape[0]


idx  = np.random.choice(trainOut.shape[0], N_train_full, replace=False)
idx2 = np.random.choice(testOut.shape[0], N_test, replace=False)

trainInFull = trainIn[idx, :]
trainOutFull = trainOut[idx]

trainIn = trainInFull[:N_train,:]
trainOut = trainOutFull[:N_train]

validIn = trainInFull[N_train:, :]
validOut = trainOutFull[N_train:]

testIn = testIn[idx2, :]
testOut = testOut[idx2]


best_configs = dict(
    max_epochs=args.epochs,
    num_hidden_layers=args.num_layers,
    num_hidden_neurons=args.hidden_size,
    weight_decay=args.weight_decay,
    learning_rate=args.learning_rate,
    optimizer=args.optimizer,
    batch_size=args.batch_size,
    activation=args.activation,
    initializer=args.weight_init,
    loss=args.loss,
)
FFNN = FeedForwardNeuralNetwork(
    num_hidden_layers=best_configs.num_hidden_layers,
    num_hidden_neurons=best_configs.num_hidden_neurons,
    X_train_raw=trainInFull,
    Y_train_raw=trainOutFull,
    N_train = N_train_full,
    X_val_raw = validIn,
    Y_val_raw = validOut,
    N_val = N_validation,
    X_test_raw = testIn,
    Y_test_raw = testOut,
    N_test = N_test,
    optimizer = best_configs.optimizer,
    batch_size = best_configs.batch_size,
    weight_decay = best_configs.weight_decay,
    learning_rate = best_configs.learning_rate,
    max_epochs = best_configs.max_epochs,
    activation = best_configs.activation,
    initializer = best_configs.initializer,
    loss = best_configs.loss
    )
FFNN.optimizer(FFNN.max_epochs, FFNN.N_train, FFNN.batch_size, FFNN.alpha)
