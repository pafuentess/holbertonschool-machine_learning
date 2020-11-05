#!/usr/bin/env python3
""" doc """

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ doc """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if type(k) != int or k <= 0:
        return None, None, None

    d = X.shape[1]
    pi = np.tile(1 / k, (k,))
    m, c = kmeans(X, k)
    s = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, s
