# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import random
import matplotlib.pyplot as plt
import numpy as np


# Calculating the sigmoid
def sigmoid(x):
    return 1 / (1.0 + math.exp(-1 * x))


# Sigmoid with Numpy
def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


# Calculating the derivative of sigmoid
def sigmoidDerivative(y):
    return y * (1 - y)


# calculating the prediction
def predict(x, y, iterations, lr, number_Of_Nodes):
    # variable declaration
    mean_Square = 0
    x_Coordinate = []
    y_Coordinate = []
    row = len(x)
    col = len(x[0])
    initial_Weights = []
    hidden_Bias = []
    hidden_Weights = []
    output_Bias = []
    newY = []

    # Initializing weights and bias between input and hidden layers
    for i in range(0, number_Of_Nodes):
        initial_Weights.append([])
        hidden_Bias.append(random.uniform(-0.5, 0.5))
        for j in range(0, col):
            initial_Weights[i].append(random.uniform(-0.5, 0.5))
    # We get two matrices initial_Weights with dimensions [number_Of_Nodes BY col] and
    # hidden_Bias with dimensions [1 BY number_Of_Nodes] both with random number between (-0.5, 0.5)
    # Initializing weights and bias between hidden and output layers
    for i in range(0, number_Of_Nodes):
        output_Bias = []
        hidden_Weights.append([])
        for j in range(0, row):
            hidden_Weights[i].append(random.uniform(-0.5, 0.5))
            output_Bias.append(random.uniform(-0.5, 0.5))
    # We get two matrices hidden_Weights with dimensions [number_Of_Nodes BY row] and
    # output_bias with dimensions [1 BY row] both with random number between (-0.5, 0.5)

    # Training the weights
    for epoch in range(0, iterations):
        newY = []
        mean_Square = 0
        for i in range(0, row):
            hidden_output = []
            newY.append([])
            # Forward Propagation from input to hidden
            for j in range(0, number_Of_Nodes):
                hidden_predict = hidden_Bias[j]
                for k in range(0, col):
                    hidden_predict += x[i][k] * initial_Weights[j][k]
                hidden_output.append(sigmoid(hidden_predict))
            output_Predict = []
            output = []
            # Forward Propagation from hidden to output
            for j in range(0, row):
                predicted = 0
                for k in range(0, number_Of_Nodes):
                    predicted += hidden_output[k] * hidden_Weights[k][j]
                predicted += output_Bias[j]
                output_Predict.append(predicted)
                output.append(sigmoid(output_Predict[j]))
                newY[i].append(sigmoid(output_Predict[j]))
            # Back propagation
            error_Without_Sigmoid = []
            error = []
            # Estimating error
            for j in range(0, row):
                error_Without_Sigmoid.append(output[j] - y[i][j])
                mean_Square += error_Without_Sigmoid[j] ** 2
                error.append(error_Without_Sigmoid[j] * sigmoidDerivative(output[j]))
            mean_Square /= row
            carry = []
            # Calculating the carry forward
            for j in range(0, number_Of_Nodes):
                reverse = 0
                for k in range(0, row):
                    reverse += error[k] * hidden_Weights[j][k]
                carry.append(reverse * sigmoidDerivative(hidden_output[j]))
            # Updating weights
            for j in range(0, row):
                output_Bias[j] -= lr * error[j]
            for j in range(0, number_Of_Nodes):
                for k in range(0, row):
                    hidden_Weights[j][k] -= lr * hidden_output[j] * error[k]
                for k in range(0, col):
                    initial_Weights[j][k] -= lr * x[i][k] * carry[j]
                hidden_Bias[j] -= lr * carry[j]
        x_Coordinate.append(epoch)
        y_Coordinate.append(mean_Square / row)

    plt.plot(x_Coordinate, y_Coordinate)
    plt.title("MSE of pattern recognition without NumPy")
    plt.xlabel("Epoch")
    plt.ylabel("MSE [Mean Square Error]")
    plt.show()
    return newY


# Prediction with Numpy
def predictNumpy(x, y, iterations, lr, number_Of_Nodes):
    # variable declaration
    row = len(x)
    col = len(x[0])
    x_Coordinate = []
    y_Coordinate = []

    # Initializing weights and biases
    weights_ih = np.random.uniform(-0.5, 0.5, (number_Of_Nodes, col))  # [numberOfNodes BY col]
    hidden_bias = np.random.uniform(-0.5, 0.5, number_Of_Nodes)  # [1 BY numberOfNodes]
    weights_oh = np.random.uniform(-0.5, 0.5, (number_Of_Nodes, row))  # [numberOfNodes BY row]
    output_bias = np.random.uniform(-0.5, 0.5, row)  # [1 BY row]
    nY = []

    # Training the weights
    for epoch in range(0, iterations):
        nY = []
        mse = 0
        for i in range(0, row):
            # Forward Propagation
            # From input to hidden layers
            hidden_output = sigmoid_np(np.dot(x[i], weights_ih.T) + hidden_bias)  # 1*25 . 25*5 + 1*5 = 1*5
            # From hidden to output layers
            output = sigmoid_np(np.dot(hidden_output, weights_oh) + output_bias)  # 1*5 . 5*10 + 1*10= 1*10
            nY.append(output)

            # Back propagation
            error_Without_Sigmoid = output - y[i]
            mse += np.sum(error_Without_Sigmoid ** 2)
            error = error_Without_Sigmoid * sigmoidDerivative(output)

            carry = np.dot(error, weights_oh.T) * sigmoidDerivative(hidden_output)

            weights_oh -= lr * np.outer(hidden_output, error)
            output_bias -= error * lr
            weights_ih -= lr * np.outer(carry, x[i].T)
            hidden_bias -= carry * lr
        x_Coordinate.append(epoch)
        y_Coordinate.append(mse / (row ** 2))

    plt.plot(x_Coordinate, y_Coordinate)
    plt.title("MSE of pattern recognition with NumPy")
    plt.xlabel("Epoch")
    plt.ylabel("MSE [Mean Square Error]")
    plt.show()
    return np.array(nY)


target = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
num_Pattern = [[0, 0, 1, 0, 0,
                0, 1, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                1, 1, 1, 1, 1,
                1, 0, 0, 0, 0,
                1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                1, 1, 1, 1, 1],
               [1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1],
               [1, 1, 1, 1, 1,
                1, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1,
                1, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
                1, 0, 0, 0, 1,
                1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1],
               [1, 1, 1, 1, 1,
                1, 0, 0, 0, 1,
                1, 1, 1, 1, 1,
                1, 0, 0, 0, 1,
                1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1,
                1, 0, 0, 0, 1,
                1, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 1],
               [1, 0, 1, 1, 1,
                1, 0, 1, 0, 1,
                1, 0, 1, 0, 1,
                1, 0, 1, 0, 1,
                1, 0, 1, 1, 1]]

changedY = predict(num_Pattern, target, 10000, 1.5, 10)

changedY_WithNumpy = predictNumpy(np.array(num_Pattern), np.array(target), 10000, 2, 10)
np.set_printoptions(precision=1, suppress= True)

print("Without Numpy")
print(np.array(changedY))
print("With Numpy")
print(changedY_WithNumpy)
