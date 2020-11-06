#!/usr/bin/env python3
""" doc """

import numpy as np


def pdf(X, m, S):
    """ doc """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[1] != S.shape[0] or X.shape[1] != S.shape[1]:
        return None

    n, d = X.shape
    X_m = X - m
    Is = np.linalg.inv(S)

    fac = np.einsum('...k,kl,...l->...', X_m, Is, X_m)
    P1 = 1. / (np.sqrt(((2 * np.pi)**d * np.linalg.det(S))))
    P2 = np.exp(-fac / 2)

    return np.maximum((P1 * P2), 1e-300)
