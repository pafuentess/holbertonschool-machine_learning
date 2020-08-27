#!/usr/bin/env python3
""" doc """

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ doc """
    m = Y.shape[1]
    keyA = "A{}".format(L)
    dAz = cache[keyA] - Y
    for i in reversed(range(1, L + 1)):
        keyA = "A{}".format(i)
        keyW = "W{}".format(i)
        keyb = "b{}".format(i)

        A = cache[keyA]
        if i == L:
            dz = dAz
        else:
            dz = dAz * (1 - (A ** 2))

        keyA = "A{}".format(i - 1)
        A = cache[keyA]

        dw = ((1 / m) * np.matmul(dz, A.T)) + ((lambtha / m) * weights[keyW])
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dAz = np.matmul(weights[keyW].T, dz)
        weights[keyW] = weights[keyW] - alpha * dw
        weights[keyb] = weights[keyb] - alpha * db
