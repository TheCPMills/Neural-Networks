import csv
import datetime
import json
import math
import matplotlib
import nnfs
import numpy
import pandas
# import pytorch
import re
import random
import scipy
import statistics
import tensorflow
import threading
import uuid

#==================================================================================================

from nnfs.datasets import spiral_data

nnfs.init()

# Dense layer
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * numpy.random.randn(n_inputs, n_neurons)
        self.biases = numpy.zeros((1, n_neurons))
    
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = numpy.dot(inputs, self.weights) + self.biases

# ReLU activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = numpy.maximum(0, inputs)

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)

# Let's see output of the first few samples:
print(activation1.output[:5])