import numpy as np
from math import pi
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Image, filedialog
import pandas as pd

### define useful parameters ###
rad = 10
width = 5
dist = 0.5

num_sample = 2500

aa = np.random.rand(2, int(num_sample//2))

radius = (rad - width / 2) + width * aa[0, :]

### for class one ###
theta1 = pi*aa[1, :] + 20*pi/180
x1 = radius*np.cos(theta1)
y1 = radius*np.sin(theta1)
label1 = 1*np.ones((1, np.size(x1)))

### for class two ###
theta2 = pi*aa[1, :] - 20*pi/180
x2 = radius*np.cos(-theta2) + rad
y2 = radius*np.sin(-theta2) - dist
label2 = -1*np.ones((1, np.size(x2)))

### adding data together ###
data = np.vstack([np.hstack([x1, x2]),
                  np.hstack([y1, y2]),
                  np.hstack([label1, label2])])

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
    print(rows)
    print(collumns)
    
    inputs = np.array([data[:,0] , data[:,1], data[:,2]])    
    print(inputs)

    #printin(data)

    return inputs

data = test()
plt.figure()
for ii in range(1, 1900):

    x = [1, data[0, ii], data[1, ii]]
    y = data[2, ii]

    if y == 1:
        plt.scatter(x[1], x[2], c='r', linewidths=2, marker=9)

    elif y == -1:
        plt.scatter(x[1], x[2], c='b', linewidths=2, marker=9)

plt.title('Ideal Sample')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()

#### Nural Net ####

def my_activation(into):
    #if into > 500:
     #   return 1
    #elif into <= -500:
     #   return -1
    #else:
    ## switch to RELU
    out = np.tanh(into) #(np.exp(into) - np.exp(-into)) / (np.exp(into) + np.exp(-into))
    return out


def derivative(into):
    
    
    out = 1 - np.square(into)
    return out


num_tr = 400
num_test = 1400

weight_int = np.random.rand(1, 3).T
weight = weight_int
iteration = 50
#eta = 1

weight = weight.T

J = np.zeros((400, 3))
e = np.zeros(num_tr)
MSE = np.zeros(iteration)

### Training ###
for ii in range(1, iteration):
    shuffle_seq = np.random.permutation(num_test)
    shuffled_data = np.array(data[:, shuffle_seq])

    for jj in range(1, num_tr):
        x = np.array([1, shuffled_data[0, jj], shuffled_data[1, jj]])
        d = np.array(shuffled_data[2, jj])
        v = np.dot(weight, x)

        y = my_activation(v)
        dy = derivative(y)
        e[jj] = d - y

        ### Gauss Newton ####
        J[jj, 0] = np.array(np.dot(-x[0], dy))
        J[jj, 1] = np.array(np.dot(-x[1], dy))
        J[jj, 2] = np.array(np.dot(-x[2], dy))

        inverse = np.linalg.inv(J.T @ J + 100 * (np.eye(3)))

        weight = weight - (inverse @ J.T @ e)

        ### Steepest Descent ###
        # weight = weight + eta * (d - y) * x


    MSE[ii] = np.sum(e**2) / iteration

### plot MSE ###
plt.figure()
plt.plot(MSE)
plt.title('Mean-Squared Error')
plt.ylabel('Cost Function value')
plt.xlabel('Iteration')
plt.grid()

### Testing ###

shuffle_seq = np.random.permutation(num_test)
shuffled_data_test = data[:, shuffle_seq]

in_put = np.zeros((3, num_test))

output_int = np.zeros(num_test)
output_opt = np.zeros(num_test)

for ii in range(1, num_test):
    in_put[:, ii] = np.array([1, shuffled_data_test[0, ii], shuffled_data_test[1, ii]])

    v_initial = np.dot(weight_int.T, in_put[:, ii])
    v_final = np.dot(weight, in_put[:, ii])

    output_int[ii] = my_activation(v_initial)
    output_opt[ii] = my_activation(v_final)

##### Display the outputs #####
### Inital Training Data ###
plt.figure()
for ii in range(1, num_test):
    x = [in_put[1, ii], in_put[2, ii]]
    y = output_int[ii]

    if y >= 0.5:
        plt.scatter(x[0], x[1], c='r', linewidths=2, marker=9)
    elif y < 0.5:
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

    if y >= 0.5:
        plt.scatter(x[0], x[1], c='r', linewidths=2, marker=9)
    elif y < 0.5:
        plt.scatter(x[0], x[1], c='b', linewidths=2, marker=9)

plt.title('Final Guess for Weights')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()

plt.show()