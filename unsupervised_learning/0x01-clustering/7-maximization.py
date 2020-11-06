#!/usr/bin/env python3
""" doc """

import numpy as np


def maximization(X, g):
    """ doc """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]
    p = np.sum(g, axis=0)
    Tp = np.ones((n,))
    if not np.isclose(p, Tp).all():
        return None, None, None
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        m[i] = np.sum((g[i, :, np.newaxis] * X), axis=0) / np.sum(g[i], axis=0)
        S[i] = np.dot(g[i] * (X - m[i]).T, (X-m[i])) / np.sum(g[i])
        pi[i] = np.sum(g[i]) / n

    return pi, m, S
