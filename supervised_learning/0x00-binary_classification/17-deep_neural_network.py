#!/usr/bin/env python3
""" doc """

import numpy as np


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
