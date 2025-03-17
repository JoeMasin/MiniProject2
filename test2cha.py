from ctypes import sizeof
from os import error
from turtle import color
import numpy as np
from math import pi
from statistics import NormalDist
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Image, filedialog
import cv2 as cv
import pandas as pd


def test():

    # initialize tkinter root window
    root = tk.Tk()
    root.withdraw()
    
    # get input image file from the user
    input_file = filedialog.askopenfilename(initialdir="./",
                                          title="Select an Input Image")
    
    data = pd.read_csv(input_file)
    data = np.array(data)
    rows, collumns = data.shape[:2]
    #print(rows)
    #print(collumns)
    
    inputs = np.array([data[:,0] , data[:,1], data[:,2]])
    #print(inputs)

    #printin(data)

    return inputs

# Define the ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Define the derivative of ReLU
def relu_derivative(x):
    return (x > 0).astype(float)

# Define the softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=0)

# Define the cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))


def main():
    # Initialize parameters
    input_size = 3  # 2 features + 1 bias
    hidden_size = 4  # Number of neurons in the hidden layer
    output_size = 2  # Number of classes

    # Initialize weights and biases randomly
    W1 = np.random.randn(hidden_size, input_size)  # Weights for input to hidden layer
    W2 = np.random.randn(output_size, hidden_size)  # Weights for hidden to output layer

    # Example input (2 features + 1 bias)
    x = test()  # Last term is the bias
    std = np.std(x, axis=0)
    mean = np.mean(x, axis=0)
    std[std == 0] = 1
    data_norm = (x - mean) / std
    
    data_norm[2,:] = x[2,:]

    x = data_norm
    y_true = np.array([data_norm[2,:], 1-data_norm[2,:]])  # One-hot encoded true label

    # Hyperparameters
    learning_rate = 0.01
    epochs = 1000
    i = 2
    # Training loop
    for epoch in range(epochs):
        for i in range(0, 500):
            # Forward pass
            # Input to hidden layer
            z1 = np.dot(W1, x[:,i])
            a1 = relu(z1)

            # Hidden layer to output layer
            z2 = np.dot(W2, a1)
            a2 = softmax(z2)

            # Compute loss
            loss = cross_entropy_loss(y_true[:,i], a2)

            # Backward pass
            # Gradient of loss w.r.t. output layer pre-activations (z2)
            dz2 = a2 - y_true[:,i]

            # Gradient of loss w.r.t. hidden layer weights (W2)
            dW2 = np.outer(dz2, a1)

            # Gradient of loss w.r.t. hidden layer activations (a1)
            da1 = np.dot(W2.T, dz2)

            # Gradient of loss w.r.t. hidden layer pre-activations (z1)
            dz1 = da1 * relu_derivative(z1)

            # Gradient of loss w.r.t. input layer weights (W1)
            dW1 = np.outer(dz1, x[:,i])

            # Update weights
            W1 -= learning_rate * dW1
            W2 -= learning_rate * dW2
            
            # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    # Final predictions
    print("Final predictions:", a2)
    print("Input: ",x[:,i])
    print(y_true[:,i])

main()