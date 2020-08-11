#!/usr/bin/env python3
""" doc """

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """ doc """
    def __init__(self, nx, layers):
        """ doc """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

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
            self.__cache[keyA] = 1 / (1 + np.exp(-z))
        return self.__cache[keyA], self.__cache

    def cost(self, Y, A):
        """ doc """
        m = Y.shape[1]
        cost = (-(1/m)) * np.sum((Y * np.log(A)) +
                                 ((1 - Y) * np.log(1.0000001 - A)))
        return (cost)

    def evaluate(self, X, Y):
        """ doc """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        return (np.round(A).astype(int), cost)

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
            dz = np.matmul(self.weights[keyW].T, dz) * (A * (1 - A))
            self.__weights[keyW] = self.__weights[keyW] - alpha * dw
            self.__weights[keyb] = self.__weights[keyb] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ doc """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        cost_store = []
        iteration_store = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.cache["A{}".format(self.__L)])
            if verbose:
                if type(step) is not int:
                    raise TypeError("step must be an integer")
                if step < 0 and step > iterations:
                    raise ValueError("step must be positive and <= iterations")
                cost_store.append(cost)
                iteration_store.append(i)
                print("Cost after {} iterations: {}".format(i, cost))
                verbose = False
                graph = False
            self.gradient_descent(Y, self.__cache, alpha)
            if ((i + 1) % step) == 0:
                verbose = True
            if i == iterations:
                graph = True
            if graph:
                if type(step) is not int:
                    raise TypeError("step must be an integer")
                if step < 0 and step > iterations:
                    raise ValueError("step must be positive and <= iterations")
                plt.plot(iteration_store, cost_store, color='blue')
                plt.title("Training Cost")
                plt.xlabel("iteration")
                plt.ylabel("cost")
                plt.show()
        return self.evaluate(X, Y)

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
        except FileNotFoundError:
            return None
