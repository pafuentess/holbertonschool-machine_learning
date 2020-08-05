#!/usr/bin/env python3
""" doc """
import numpy as np


class Neuron:
    """ doc """
    def __init__(self, nx):
        """ doc """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ doc """
        return self.__W

    @property
    def b(self):
        """ doc """
        return self.__b

    @property
    def A(self):
        """ doc """
        return self.__A

    def forward_prop(self, X):
        """ doc """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """ doc """
        m = Y.shape[1]
        cost = (-(1/m)) * np.sum((Y * np.log(A)) +
                                 ((1 - Y) * np.log(1.0000001 - A)))
        return (cost)
