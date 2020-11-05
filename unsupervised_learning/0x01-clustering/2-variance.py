#!/usr/bin/env python3
""" doc """

import numpy as np


def variance(X, C):
    """ doc """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray):
        return None

    try:
        distance = np.sqrt(((X - C[:, np.newaxis]) ** 2).sum(axis=-1))
        min_ = np.min(distance, axis=0)
        Variance = np.sum(min_ ** 2)
        return Variance
    except Exception:
        return None
