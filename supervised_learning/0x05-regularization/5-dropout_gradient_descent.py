#!/usr/bin/env python3
""" doc """

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ doc """
    m = Y.shape[1]
    keyA = "A{}".format(L)
    dAz = cache[keyA] - Y
    for i in range(L, 0, -1):
        keyW = "W{}".format(i)
        keyb = "b{}".format(i)
        keyd = "D{}".format(i)

        A = cache[keyA]
        if i == L:
            dz = dAz
        else:
            dz = dAz * (1 - ((cache[keyA]) ** 2)) * (cache[keyd] / keep_prob)

        keyA = "A{}".format(i - 1)
        A = cache[keyA]
        dw = (1 / m) * np.matmul(dz, A.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dAz = np.matmul(weights[keyW].T, dz)
        weights[keyW] = weights[keyW] - alpha * dw
        weights[keyb] = weights[keyb] - alpha * db
