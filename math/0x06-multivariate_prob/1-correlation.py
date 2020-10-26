#!/usr/bin/env python3
""" doc """

import numpy as np


def correlation(C):
    """ doc """
    if type(C) != np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if (C.shape[0] != C.shape[1] or len(C.shape) != 2):
        raise ValueError("C must be a 2D square matrix")

    var = np.diag(1 / np.sqrt(np.diag(C)))
    corr = np.matmul(np.matmul(var, C), var)

    return corr
