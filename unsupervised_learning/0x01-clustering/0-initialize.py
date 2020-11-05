#!/usr/bin/env python3
""" doc """
import numpy as np


def initialize(X, k):
    """ doc """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None

    minimum = np.amin(X, axis=0)
    maximum = np.amax(X, axis=0)

    return np.random.uniform(minimum, maximum, (k, X.shape[1]))
