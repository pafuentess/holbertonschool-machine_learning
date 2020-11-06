#!/usr/bin/env python3
""" doc """

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ doc """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) != float or tol <= 0:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None

    P_likehood = 0
    pi, m, S = initialize(X, k)
    g, likehood = expectation(X, pi, m, S)
    text = 'Log Likelihood after {} iterations: {}'

    for i in range(iterations):
        if verbose and (i % 10 == 0):
            print(text.format(i, likehood.round(5)))
        pi, m, S = maximization(X, g)
        g, likehood = expectation(X, pi, m, S)

        if abs(likehood - P_likehood) <= tol:
            break
        P_likehood = likehood

    if verbose:
        print(text.format(i + 1, likehood.round(5)))

    return pi, m, S, g, likehood
