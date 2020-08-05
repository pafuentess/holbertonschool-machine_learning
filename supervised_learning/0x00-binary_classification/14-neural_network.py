#!/usr/bin/env python3
""" doc """
import numpy as np


class NeuralNetwork:
    """ doc """
    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ doc """
        return self.__W1

    @property
    def b1(self):
        """ doc """
        return self.__b1

    @property
    def A1(self):
        """ doc """
        return self.__A1

    @property
    def W2(self):
        """ doc """
        return self.__W2

    @property
    def b2(self):
        """ doc """
        return self.__b2

    @property
    def A2(self):
        """ doc """
        return self.__A2

    def forward_prop(self, X):
        """ doc """
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = (1 / (1 + np.exp(-z2)))
        return(self.__A1, self.__A2)

    def cost(self, Y, A):
        """ doc """
        m = Y.shape[1]
        cost = (-(1/m)) * np.sum((Y * np.log(A)) +
                                 ((1 - Y) * np.log(1.0000001 - A)))
        return (cost)

    def evaluate(self, X, Y):
        """ doc """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        return (np.round(A2).astype(int), cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        m = X.shape[1]
        dz2 = A2 - Y
        dw2 = (1/m) * np.matmul(A1, dz2.T)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
        self.__W2 = self.__W2 - (alpha*dw2).T
        self.__b2 = self.__b2 - (alpha * db2)
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1 / m) * np.matmul(X, dz1.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
        self.__W1 = self.__W1 - (alpha * dw1).T
        self.__b1 = self.__b1 - (alpha * db1)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ doc """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        return self.evaluate(X, Y)
