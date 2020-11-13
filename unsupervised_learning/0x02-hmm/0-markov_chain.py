#!/usr/bin/env python3
""" doc """

import numpy as np


def markov_chain(P, s, t=1):
    """ doc """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or len(s.shape) != 2:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[0]:
        return None
    if not isinstance(t, int) or t < 1:
        return None

    Pk = np.linalg.matrix_power(P, t)
    return np.matmul(s, Pk)
