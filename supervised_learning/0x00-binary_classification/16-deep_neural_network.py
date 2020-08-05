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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        keyw = ""
        keyb = ""
        for i in range(self.L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            b = np.zeros((layers[i], 1))
            keyw = "W{}".format(i + 1)
            keyb = "b{}".format(i + 1)
            self.weights[keyw] = w
            self.weights[keyb] = b
            nx = layers[i]
