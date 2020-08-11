#!/usr/bin/env python3
""" doc """

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """ doc """
    def __init__(self, nx, layers, activation='sig'):
        """ doc """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        keyw = ""
        keyb = ""
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            b = np.zeros((layers[i], 1))
            keyw = "W{}".format(i + 1)
            keyb = "b{}".format(i + 1)
            self.__weights[keyw] = w
            self.__weights[keyb] = b
            nx = layers[i]

    @property
    def L(self):
        """ doc """
        return self.__L

    @property
    def weights(self):
        """ doc """
        return self.__weights

    @property
    def cache(self):
        """ doc """
        return self.__cache

    @property
    def activation(self):
        """ doc """
        return self.__activation

    def forward_prop(self, X):
        """ doc """
        self.__cache["A0"] = X
        for i in range(self.__L):
            keyW = "W{}".format(i + 1)
            keyb = "b{}".format(i + 1)
            keyA = "A{}".format(i)
            b = self.__weights[keyb]
            z = np.matmul(self.__weights[keyW], self.__cache[keyA]) + b
            keyA = "A{}".format(i + 1)
            if i != self.__L - 1:
                if self.__activation == 'sig':
                    self.__cache[keyA] = 1 / (1 + np.exp(-z))
                elif self.__activation == 'tanh':
                    self.__cache[keyA] = np.sinh(z) / np.cosh(z)
            else:
                t = np.exp(z)
                activation = np.exp(z) / np.sum(t, axis=0, keepdims=True)
                self.__cache[keyA] = activation
        return self.__cache[keyA], self.__cache

    def cost(self, Y, A):
        """ doc """
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(A), axis=1, keepdims=True)
        cost = np.sum(loss) / m
        return (cost)

    def evaluate(self, X, Y):
        """ doc """
        A, cache = self.forward_prop(X)
        prediction = np.where(A == np.amax(A, axis=0), 1, 0)
        cost = self.cost(Y, A)
        return (prediction, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ doc """
        m = Y.shape[1]
        keyA = "A{}".format(self.__L)
        dz = cache[keyA] - Y
        for i in range(self.__L, 0, -1):
            keyA = "A{}".format(i - 1)
            keyW = "W{}".format(i)
            keyb = "b{}".format(i)
            A = cache[keyA]
            dw = (1 / m) * np.matmul(dz, A.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            self.__weights[keyW] = self.__weights[keyW] - alpha * dw
            self.__weights[keyb] = self.__weights[keyb] - alpha * db
            if self.__activation == 'sig':
                dz = np.matmul(self.__weights[keyW].T, dz) * (A * (1 - A))
            elif self.__activation == 'tanh':
                dz = np.matmul(self.__weights[keyW].T, dz) * (1 - A * A)

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """trains the model"""
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        _, cache = self.forward_prop(X)
        cost_list = []
        iter_x = []
        for i in range(iterations + 1):
            print(i)
            A, cost = self.evaluate(X, Y)
            if verbose is True and (
                    i % step == 0 or i == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, cost))
                cost_list.append(cost)
                iter_x.append(i)
            if i != iterations:
                self.gradient_descent(Y, cache, alpha)
                _, cache = self.forward_prop(X)
        if graph is True:
            plt.plot(iter_x, cost_list)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return (A, cost)

    def save(self, filename):
        """ doc """
        extension = filename.split(".")
        if len(extension) == 1:
            filename = filename + ".pkl"

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """ doc """
        try:
            with open(filename, 'rb') as f:
                fileObject = pickle.load(f)
            return fileObject
        except (OSError, IOError) as errors:
            return None
