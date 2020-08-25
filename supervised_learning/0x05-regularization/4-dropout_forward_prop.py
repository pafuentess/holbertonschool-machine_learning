#!/usr/bin/env python3
""" doc """

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ doc """
    cache = {}
    cache["A0"] = X
    for i in range(L):
        keyW = "W{}".format(i + 1)
        keyb = "b{}".format(i + 1)
        keyd3 = "D{}".format(i + 1)
        keyA = "A{}".format(i)
        b = weights[keyb]
        z = np.matmul(weights[keyW], cache[keyA]) + b
        if i != L - 1:
            a = np.sinh(z) / np.cosh(z)
            d3 = np.random.binomial(1, keep_prob, (a.shape[0], a.shape[1]))
            cache[keyd3] = d3
            a = np.multiply(a, d3)
            keyA = "A{}".format(i + 1)
            cache[keyA] = a / keep_prob
        else:
            t = np.exp(z)
            keyA = "A{}".format(i + 1)
            cache[keyA] = t / np.sum(t, axis=0, keepdims=True)
    return cache
