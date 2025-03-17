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



def printin(input):
    row,col = input.shape[:2]
    plt.figure()
    for i in range(0, row):
        if input[i,2] == 1:
            plt.scatter(input[i,0], input[i,1], color='blue')
        else:
            plt.scatter(input[i,0], input[i,1], color='red')

    plt.title('Ideal Sample')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.grid()
    plt.show()


def activation(into):
    ## switch to RELU
    out = np.tanh(into) #(np.exp(into) - np.exp(-into)) / (np.exp(into) + np.exp(-into))
    return out


def derivative(into):

    out = 1 - np.square(into)
    return out


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


def main():
    #### Basic 3 neuron Nural Net with 1 hidden layer ####
    ### define peramiters ###

    data = test()
    data_rows, data_cols = data.shape[:2]

    #### Z-score normalize ####

    std = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    std[std == 0] = 1
    data_norm = (data - mean) / std

    data_norm[2, :] = data[2, :]
    ### define the number of train and test data being used ###
    max_data_size = max(data_rows, (data_cols - 1))

    num_tr = (max(data_rows, (data_cols - 1))) // 4
    num_test = max_data_size - (num_tr + 1)

    # He initialization for ReLU
    weight_1 = np.random.randn(3, 3) * np.sqrt(2.0 / 3)
    weight_2 = np.random.randn(2, 4) * np.sqrt(2.0 / 4)
    # weight_2 = np.random.normal(weight_2)

    iterations = 500
    # eta = 1 ### would be used if we were using epocs but we are not because we willuse all the data in 1 run of training and testing ###
    learning_rate = 0.01
    # J = np.zeros((500, 3))
    # error = np.zeros(num_tr)
    err = np.zeros(iterations)
    MSE = np.zeros(iterations)
    # loss = np.zeros(num_tr)
    outguess = np.zeros((3, num_tr))
    total_L = 0

    ### Training ###
    for i in range(0, iterations):
        shuffle_seq = np.random.permutation(num_test)
        shuffled_data = np.array(data_norm[:, shuffle_seq])
        shuffled_data_location = np.array(data[:, shuffle_seq])
        
        total_loss = 0  # Track total loss for averaging
        
        for j in range(0, num_tr):
            #### Forward Propagation ####
            x = np.array([shuffled_data[0, j], shuffled_data[1, j], 1])
            d = np.array([shuffled_data[2, j], 1 - shuffled_data[2, j]])
            
            z = np.dot(weight_1.T, x)
            h = np.maximum(0, z)  # ReLU activation
            h = np.append(h, 1)  # Add bias term
            
            y = np.dot(weight_2, h)
            
            ## SoftMax of output ##
            exp_y = np.exp(y - np.max(y))  # Numerical stability
            y_soft = exp_y / np.sum(exp_y)
            
            outguess[:, j] = np.array([shuffled_data_location[0, j], shuffled_data_location[1, j], y_soft[0]])
            
            ## Cross Entropy loss ## 
            L = -np.sum(d * np.log(y_soft + 1e-10))  # Add small epsilon to avoid log(0)
            total_loss += L
            
            #### Backward Propagation ####
            output_error = y_soft - d
            weight_2_updates = np.outer(output_error, h)
            
            hidden_error = np.dot(weight_2.T, output_error) * (h > 0)
            weight_1_updates = np.outer(hidden_error[:-1], x)

            # Print key variables for debugging
            if i % 100 == 0 and j == 0:
                print(f"Iteration {i}, Sample {j}")
                print("x:", x)
                print("d:", d)
                print("z:", z)
                print("h:", h)
                print("y:", y)
                print("y_soft:", y_soft)
                print("output_error:", output_error)
                print("hidden_error:", hidden_error)
                print("weight_1_updates:", weight_1_updates)
                print("weight_2_updates:", weight_2_updates)
                print("Loss:", L)
                print()

            # Update weights
            weight_1 = weight_1 - learning_rate * weight_1_updates
            weight_2 = weight_2 - learning_rate * weight_2_updates
        
        MSE[i] = total_loss / num_tr  # Average loss

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {MSE[i]}")

        if i % 100 == 0:
            plt.figure()
            for ii in range(1, num_tr):
                x = [outguess[0, ii], outguess[1, ii]]
                y = outguess[2, ii]

                if y >= 0.5:
                    plt.scatter(x[0], x[1], c="r", linewidths=2, marker=9)
                elif y < 0.5:
                    plt.scatter(x[0], x[1], c="b", linewidths=2, marker=9)

            plt.title("Final Guess for Weights")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.grid()

            plt.show()

    #        MSE[i] = L/iterations #np.sum(error**2) / iterations

    ### plot MSE ###
    plt.figure()
    plt.plot(MSE, color="red")
    plt.plot(err, color="green")
    plt.title('Mean-Squared Error')
    plt.ylabel('Cost Function value')
    plt.xlabel('Iteration')
    plt.grid()
    plt.show()

    
    '''### Testing ###

    shuffle_seq = np.random.permutation(num_test)
    shuffled_data_test = data[:, shuffle_seq]

    in_put = np.zeros((3, num_test))

    output_int = np.zeros(num_test)
    output_opt = np.zeros(num_test)

    for ii in range(1, num_test):
        in_put[:, ii] = np.array([1, shuffled_data_test[0, ii], shuffled_data_test[1, ii]])

        v_initial = np.dot(weight_int.T, in_put[:, ii])
        v_final = np.dot(weight, in_put[:, ii])

        output_int[ii] = max(v_initial)
        output_opt[ii] = max(v_final)

    ##### Display the outputs #####
    ### Inital Training Data ###
    plt.figure()
    for ii in range(1, num_test):
        x = [in_put[1, ii], in_put[2, ii]]
        y = output_int[ii]

        if y >= 0:
            plt.scatter(x[0], x[1], c='r', linewidths=2, marker=9)
        elif y < 0:
            plt.scatter(x[0], x[1], c='b', linewidths=2, marker=9)

    plt.title('Initial Guess for Weights')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()

    ### Final Output ###
    plt.figure()
    for ii in range(1, num_test):
        x = [in_put[1, ii], in_put[2, ii]]
        y = output_opt[ii]

        if y >= 0:
            plt.scatter(x[0], x[1], c='r', linewidths=2, marker=9)
        elif y < 0:
            plt.scatter(x[0], x[1], c='b', linewidths=2, marker=9)

    plt.title('Final Guess for Weights')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid()

    plt.show()'''
    


main()