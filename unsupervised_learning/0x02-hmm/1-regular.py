#!/usr/bin/env python3
""" doc """

import numpy as np


def regular(P):
    """ doc """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    S = np.full(n, (1 / n))[np.newaxis, ...]
    S_prev = S
    K = 1
    while 1:
        Pk = np.linalg.matrix_power(P, K)
        if np.any(Pk <= 0):
            return None
        S = np.matmul(S, P)
        if np.all(S_prev == S):
            return S
        S_prev = S
        K += 1
