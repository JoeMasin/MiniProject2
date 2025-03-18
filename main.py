import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import pandas as pd

def date_reader():
    # Initialize tkinter and open file dialog to select input file
    root = tk.Tk()
    root.withdraw()
    input_file = filedialog.askopenfilename(initialdir="./", title="Select an Input Image")
    data = pd.read_csv(input_file)
    data = np.array(data)
    return data

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=0)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-10))  # Add small epsilon to avoid log(0)

def main():
    # Initialize parameters
    input_size = 2  # features
    hidden_size = 3  # Number of neurons in the hidden layer
    output_size = 2  # Number of classes

    # Initialize weights
    W1 = np.random.randn(hidden_size, input_size + 1) * np.sqrt(2.0 / (input_size + 1))
    W2 = np.random.randn(output_size, hidden_size + 1) * np.sqrt(2.0 / (hidden_size + 1))

    data = date_reader()
    data_rows, data_cols = data.shape

    # Z-score normalization 
    std = np.std(data[:, :2], axis=0)
    mean = np.mean(data[:, :2], axis=0)
    std[std == 0] = 1
    data_norm = (data[:, :2] - mean) / std
    data_norm = np.column_stack((data_norm, data[:, 2]))  # Keep labels as they are

    # One-hot encode labels
    y_true = np.array([data[:, 2], 1 - data[:, 2]])

    plt.figure()
    for i in range(data_rows):
        if data[i, 2] >= 0.5:
            plt.scatter(data[i, 0], data[i, 1], c='b', marker='o', label='Class 1' if i == 0 else "")
        else:
            plt.scatter(data[i, 0], data[i, 1], c='r', marker='o', label='Class 0' if i == 0 else "")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Original Data")
    plt.legend()
    plt.show()
    
    plt.figure()
    for i in range(data_rows):
        input_with_bias = np.append(data_norm[i, :2], 1)
        z1 = np.dot(W1, input_with_bias)
        h = relu(z1)
        h_with_bias = np.append(h, 1)
        z2 = np.dot(W2, h_with_bias)
        y_soft = softmax(z2)

        if y_soft[0] >= 0.5:
            plt.scatter(data[i, 0], data[i, 1], c='b', marker='o', label='Class 1' if i == 0 else "")
        else:
            plt.scatter(data[i, 0], data[i, 1], c='r', marker='o', label='Class 0' if i == 0 else "")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Original Weights")
    plt.legend()
    plt.show()
    
    # Hyperparameters
    lr = 0.01  # Learning rate
    epochs = 20  # Number of epochs

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(data_rows):
            # Forward pass
            input_with_bias = np.append(data_norm[i, :2], 1)  
            z1 = np.dot(W1, input_with_bias)
            h = relu(z1)
            h_with_bias = np.append(h, 1) 
            z2 = np.dot(W2, h_with_bias)
            y = softmax(z2)

            # Compute loss
            loss = cross_entropy_loss(y_true[:, i], y)
            epoch_loss += loss

            # Backward pass
            dz2 = y - y_true[:, i]
            dW2 = np.outer(dz2, h_with_bias)
            dh = np.dot(W2[:, :-1].T, dz2)  
            dz1 = dh * relu_derivative(z1)
            dW1 = np.outer(dz1, input_with_bias)

            # Update weights
            W1 -= lr * dW1
            W2 -= lr * dW2

        # Print average loss for the epoch
        print(f"Epoch {epoch}, Loss: {epoch_loss / data_rows}")

    # Final predictions and plotting
    plt.figure()
    for i in range(data_rows):
        input_with_bias = np.append(data_norm[i, :2], 1)
        z1 = np.dot(W1, input_with_bias)
        h = relu(z1)
        h_with_bias = np.append(h, 1)
        z2 = np.dot(W2, h_with_bias)
        y_soft = softmax(z2)

        if y_soft[0] >= 0.5:
            plt.scatter(data[i, 0], data[i, 1], c='b', marker='o', label='Class 1' if i == 0 else "")
        else:
            plt.scatter(data[i, 0], data[i, 1], c='r', marker='o', label='Class 0' if i == 0 else "")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Final weights")
    plt.legend()
    plt.show()

    print("Training complete")
    
    ### Found a pretty cool visualization of the decision boundry online
    # Create a 100x100 grid to visualize the decision boundary
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Normalize the grid 
    grid_norm = (grid - mean) / std

    # Predict class probabilities for each point on the grid
    predictions = []
    for point in grid_norm:
        input_with_bias = np.append(point, 1)  
        z1 = np.dot(W1, input_with_bias)
        h = relu(z1)
        h_with_bias = np.append(h, 1)  
        z2 = np.dot(W2, h_with_bias)
        y = softmax(z2)
        predictions.append(y[0])  # Probability of class 1

    predictions = np.array(predictions).reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, predictions, levels=50, cmap="RdBu", alpha=0.8)
    plt.colorbar(label="Probability of Class 1")
    #plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap="RdBu", edgecolors="k", label="Data Points")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.legend()
    plt.show()

main()