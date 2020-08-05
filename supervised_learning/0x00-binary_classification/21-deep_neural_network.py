#!/usr/bin/env python3
""" doc"""

import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """ doc """
    def __init__(self, nx, layers):
        """ doc """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        keyw = ""
        keyb = ""
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
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
        key_final_A = "A{}".format(self.__L)
        Af = cache[key_final_A]
        dAf = (-1 * (Y / Af)) + (1 - Y)/(1-Af)

        for i in reversed(range(1, self.__L + 1)):
            keyW = "W{}".format(i)
            keyb = "b{}".format(i)
            key_final_A = "A{}".format(i)
            key_final_A1 = "A{}".format(i - 1)
            Af = cache[key_final_A]
            Al1 = cache[key_final_A1]
            dsigmoid = Af * (1 - Af)
            dz = np.multiply(dAf, dsigmoid)
            dw = (1 / m) * np.matmul(dz, Al1.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            w = self.__weights[keyW]
            dA = np.matmul(w.T, dz)
            self.__weights[keyW] = self.__weights[keyW] - alpha * dw
            self.__weights[keyb] = self.__weights[keyb] - alpha * db
