#!/usr/bin/env python3
"""Creating a deep neural network"""

import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network"""
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not all(isinstance(l, int) and l > 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                self.__weights['W1'] = np.random.randn(layers[0], nx) * np.sqrt(2 / nx)
            else:
                self.__weights[f'W{i + 1}'] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights[f'b{i + 1}'] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Number of layers in the neural network"""
        return self.__L

    @property
    def cache(self):
        """Intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """Holds all weights and biases"""
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation"""
        self.__cache["A0"] = X
        for i in range(1, self.L + 1):
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']
            A_prev = self.__cache[f'A{i - 1}']
            Z = np.matmul(W, A_prev) + b
            self.__cache[f'A{i}'] = 1 / (1 + np.exp(-Z))
        return self.__cache[f'A{self.L}'], self.__cache

    def cost(self, Y, A):
        """Calculate cost using cross-entropy loss"""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        """Evaluate the network's predictions"""
        A_final, _ = self.forward_prop(X)
        predictions = np.where(A_final >= 0.5, 1, 0)
        cost = self.cost(Y, A_final)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient descent for updating weights and biases"""
        m = Y.shape[1]
        A_final = cache[f"A{self.__L}"]
        da = A_final - Y  # For the last layer

        for i in range(self.L, 0, -1):
            A_prev = cache[f"A{i - 1}"]
            W = self.__weights[f"W{i}"]
            b = self.__weights[f"b{i}"]

            dz = da
            db = np.sum(dz, axis=1, keepdims=True) / m
            dw = np.matmul(dz, A_prev.T) / m
            if i > 1:  # Backpropagate error
                da = np.matmul(W.T, dz) * A_prev * (1 - A_prev)

            self.__weights[f"W{i}"] -= alpha * dw
            self.__weights[f"b{i}"] -= alpha * db
