#!/usr/bin/env python3
""" doc """

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ doc """
    m = Y.shape[1]
    keyA = "A{}".format(L)
    dz = cache[keyA] - Y
    for i in range(L, 0, -1):
        keyA = "A{}".format(i - 1)
        keyW = "W{}".format(i)
        keyb = "b{}".format(i)
        A = cache[keyA]
        dw = ((1 / m) * np.matmul(dz, A.T)) + ((lambtha / m) * weights[keyW])
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dz = np.matmul(weights[keyW].T, dz) * (A * (1 - A))
        weights[keyW] = weights[keyW] - alpha * dw
        weights[keyb] = weights[keyb] - alpha * db
