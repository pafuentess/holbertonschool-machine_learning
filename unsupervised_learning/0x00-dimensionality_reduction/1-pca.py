#!/usr/bin/env python3
""" doc """

import numpy as np


def pca(X, ndim):
    """ doc """
    x = X - np.mean(X, axis=0)
    u, sigma, v = np.linalg.svd(x)
    w = v[:ndim].T

    return np.matmul(x, w)
