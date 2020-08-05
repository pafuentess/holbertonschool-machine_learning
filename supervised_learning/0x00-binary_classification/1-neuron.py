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
