#!/usr/bin/env python3
""" doc """

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ doc """
    if type(X) is not np.ndarray or type(m) is not np.ndarray:
        return (None, None)
    if type(S) is not np.ndarray or type(pi) is not np.ndarray:
        return (None, None)
    if len(X.shape) != 2 or len(S.shape) != 3:
        return (None, None)
    if len(pi.shape) != 1 or len(m.shape) != 2:
        return (None, None)
    if m.shape[1] != X.shape[1]:
        return (None, None)
    if S.shape[2] != S.shape[1]:
        return (None, None)
    if S.shape[0] != pi.shape[0] or S.shape[0] != m.shape[0]:
        return (None, None)
    if np.min(pi) < 0:
        return (None, None)

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
