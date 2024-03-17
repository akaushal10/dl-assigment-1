import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import argparse
from feedForwardNeuralNetwork import FeedForwardNeuralNetwork
from constant import SIGMOID_KEY,TANH_KEY,RELU_KEY,XAVIER_KEY,RANDOM_KEY,SGD_KEY,MGD_KEY,NAG_KEY,RMSPROP_KEY,ADAM_KEY,NADAM_KEY,CROSS_ENTROPY_KEY,MEAN_SQUARE_KEY
from constant import FASHION_MNIST_DATASET_KEY,MNIST_DATASET_KEY

parser = argparse.ArgumentParser()
parser.add_argument("-wp","--wandb_project",help="Project name used to track experiments in Weights & Biases dashboard",default="dl-assigment-1")
parser.add_argument("-we","--wandb_entity",help="Wandb Entity used to track experiments in the Weights & Biases dashboard.",default="cs23m007")
parser.add_argument("-d","--dataset",help=f"choices: [{FASHION_MNIST_DATASET_KEY}, {MNIST_DATASET_KEY}]",choices=[FASHION_MNIST_DATASET_KEY,MNIST_DATASET_KEY],default=FASHION_MNIST_DATASET_KEY)
parser.add_argument("-e","--epochs",help="Number of epochs to train neural network.",choices=['5','10'],default=10)
parser.add_argument("-b","--batch_size",help="Batch size used to train neural network.",choices=['16','32','64'],default=32)
parser.add_argument("-l","--loss",help=f"choices: [{CROSS_ENTROPY_KEY}, {MEAN_SQUARE_KEY}]",choices=[CROSS_ENTROPY_KEY,MEAN_SQUARE_KEY],default=CROSS_ENTROPY_KEY)
parser.add_argument("-o","--optimizer",help=f"choices: [{SGD_KEY}, {MGD_KEY}, {NAG_KEY}, {RMSPROP_KEY}, {ADAM_KEY}, {NADAM_KEY}]",choices=[SGD_KEY,MGD_KEY,NAG_KEY,RMSPROP_KEY,ADAM_KEY,NADAM_KEY],default=NADAM_KEY)
parser.add_argument("-lr","--learning_rate",help="Learning rate used to optimize model parameters",choices=['1e-3','1e-4'],default=1e-3)
parser.add_argument("-m","--momentum",help="Momentum used by momentum and nag optimizers.",choices=['0.5'],default=0.5)
parser.add_argument("-beta","--beta",help="Beta used by rmsprop optimizer",choices=['0.5'],default=0.5)
parser.add_argument("-beta1","--beta1",help="Beta1 used by adam and nadam optimizers.",choices=['0.5'],default=0.5)
parser.add_argument("-beta2","--beta2",help="Beta2 used by adam and nadam optimizers.",choices=['0.5'],default=0.5)
parser.add_argument("-eps","--epsilon",help="Epsilon used by optimizers.",choices=['0.000001'],default=0.000001)
parser.add_argument("-w_d","--weight_decay",help="Weight decay used by optimizers.",choices=['0','0.0005','0.5'],default=0)
parser.add_argument("-w_i","--weight_init",help=f"choices: [{RANDOM_KEY}, {XAVIER_KEY}]",choices=[RANDOM_KEY,XAVIER_KEY],default=XAVIER_KEY)
parser.add_argument("-nhl","--num_layers",help="Number of hidden layers used in feedforward neural network.",choices=['3','4','5'],default=3)
parser.add_argument("-sz","--hidden_size",help="Number of hidden neurons in a feedforward layer.",choices=['32','64','128'],default=128)
parser.add_argument("-a","--activation",help=f"choices: [{SIGMOID_KEY}, {TANH_KEY}, {RELU_KEY}]",choices=[SIGMOID_KEY,TANH_KEY,RELU_KEY],default=RELU_KEY)
args = parser.parse_args()

(trainIn, trainOut), (testIn, testOut) = fashion_mnist.load_data()
if(args.dataset==MNIST_DATASET_KEY):
    (trainIn, trainOut), (testIn, testOut) = mnist.load_data()

if type(args.learning_rate)==type(''):
    args.learning_rate = float(args.learning_rate)
if(type(args.beta)==type('')):
    args.beta = float(args.beta)
if(type(args.beta1)==type('')):
    args.beta1 = float(args.beta1)
if(type(args.beta2)==type('')):
    args.beta2 = float(args.beta2)
if(type(args.weight_decay)==type('')):
    args.weight_decay = float(args.weight_decay)
if(type(args.batch_size)==type('')):
    args.batch_size = int(args.batch_size)
if(type(args.epochs)==type('')):
    args.epochs = float(args.epochs)
if(type(args.epsilon)==type('')):
    args.epsilon = float(args.epsilon)
if(type(args.hidden_size)==type('')):
    args.hidden_size = int(args.hidden_size)
if(type(args.num_layers)==type('')):
    args.num_layers = int(args.num_layers)


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
if(type(best_configs["max_epochs"])==type('')):
    best_configs["max_epochs"] = int(best_configs["max_epochs"])
if(type(best_configs["num_hidden_layers"])==type('')):
    best_configs["num_hidden_layers"] = int(best_configs["num_hidden_layers"])
if(type(best_configs["num_hidden_neurons"])==type('')):
    best_configs["num_hidden_neurons"] = int(best_configs["num_hidden_neurons"])

if(type(best_configs["weight_decay"])==type('')):
    best_configs["weight_decay"] = float(best_configs["weight_decay"])
if(type(best_configs["learning_rate"])==type('')):
    best_configs["learning_rate"] = float(best_configs["learning_rate"])
if(type(best_configs["batch_size"])==type('')):
    best_configs["batch_size"] = int(best_configs["batch_size"])


FFNN = FeedForwardNeuralNetwork(
    num_hidden_layers=best_configs["num_hidden_layers"],
    num_hidden_neurons=best_configs["num_hidden_neurons"],
    X_train_raw=trainInFull,
    Y_train_raw=trainOutFull,
    N_train = N_train_full,
    X_val_raw = validIn,
    Y_val_raw = validOut,
    N_val = N_validation,
    X_test_raw = testIn,
    Y_test_raw = testOut,
    N_test = N_test,
    optimizer = best_configs["optimizer"],
    batch_size = best_configs["batch_size"],
    weight_decay = best_configs["weight_decay"],
    learning_rate = best_configs["learning_rate"],
    max_epochs = best_configs["max_epochs"],
    activation = best_configs["activation"],
    initializer = best_configs["initializer"],
    loss = best_configs["loss"]
    )
FFNN.optimizer(FFNN.max_epochs, FFNN.N_train, FFNN.batch_size, FFNN.alpha)
