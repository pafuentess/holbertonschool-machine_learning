#!/usr/bin/env python3
""" doc """

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ doc """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if X.shape[1] != S.shape[1] or S.shape[1] != S.shape[2]:
        return None, None
    if X.shape[1] != m.shape[1] or m.shape[0] != S.shape[0]:
        return None, None
    if pi.shape[0] != m.shape[0]:
        return None, None

    n, d = X.shape
    k = S.shape[0]
    zeros = np.zeros((k, n))

    for i in range(k):
        p = pdf(X, m[i], S[i])
        prior = pi[i]
        zeros[i] = prior * p
    g = zeros / np.sum(zeros, axis=0)
    likehood = np.sum(np.log(np.sum(zeros, axis=0)))

    return g, likehood
