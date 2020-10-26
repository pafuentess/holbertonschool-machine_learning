#!/usr/bin/env python3
""" doc """

import numpy as np


def mean_cov(X):
    """ doc """
    if type(X) != np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if (X.shape[0] < 2):
        raise ValueError("X must contain multiple data points")

    n = X.shape[0]
    mean = np.mean(X, axis=0)
    d = X.shape[1]
    mean = mean.reshape(1, d)

    X = X - mean

    cov = (np.matmul(X.T, X) / (n - 1))

    return (mean, cov)
