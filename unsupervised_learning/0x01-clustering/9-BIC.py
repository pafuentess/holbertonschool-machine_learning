#!/usr/bin/env python3
""" doc """

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """ doc """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None
    if not isinstance(kmax, int) or kmax <= 0 or kmax >= X.shape[0]:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol <= 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    K = []
    R = []
    li = []
    b = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, likehood = expectation_maximization(X, k, iterations,
                                                         tol, verbose)
        K.append(k)
        R.append((pi, m, S))
        li.append(likehood)
        p = (k * d * (d + 1) / 2) + (d * k) + k - 1
        BIC = p * np.log(n) - 2 * likehood
        b.append(BIC)
    b = np.array(b)
    best = np.argmin(b)
    li = np.array(li)

    return K[best], R[best], li, b
