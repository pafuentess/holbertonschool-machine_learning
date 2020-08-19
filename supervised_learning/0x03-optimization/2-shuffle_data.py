#!/usr/bin/env python3
""" doc """
import numpy as np


def shuffle_data(X, Y):
    """ doc """
    permutation = np.random.permutation(len(Y))
    Xp = X[permutation]
    Yp = Y[permutation]
    return (Xp, Yp)
