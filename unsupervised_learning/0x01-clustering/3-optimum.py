#!/usr/bin/env python3
""" doc """

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ doc """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if type(kmin) != int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None

    if type(kmax) != int or kmax <= 0 or kmin >= X.shape[0]:
        return None, None

    if kmin >= kmax:
        return None, None

    if type(iterations) != int or iterations <= 0:
        return None, None

    results = []
    Vars = []

    for k in range(kmin, kmax + 1):
        centroid, labels = kmeans(X, k, iterations)
        results.append((centroid, labels))
        if k == kmin:
            var = variance(X, centroid)
        Vars.append(var - variance(X, centroid))

    return (results, Vars)
