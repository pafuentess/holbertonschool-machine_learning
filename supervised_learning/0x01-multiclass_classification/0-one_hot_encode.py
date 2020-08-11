#!/usr/bin/env python3
""" doc """

import numpy as np


def one_hot_encode(Y, classes):
    """ doc """
    if type(Y) is not np.ndarray:
        return None
    if len(Y) == 0:
        return None
    if type(classes) is not int or classes <= Y.max():
        return None
    length = Y.shape[0]
    matrix = np.zeros((classes, length))
    for c, m in enumerate(Y):
        matrix[m][c] = 1
    return (matrix)
