#!/usr/bin/env python3
""" doc """

import numpy as np


def definiteness(matrix):
    """ doc """
    if not isinstance(matrix, np.ndarray):
        raise TypeError('matrix must be a numpy.ndarray')

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    T = np.transpose(matrix)
    if not np.array_equal(T, matrix):
        return None

    Ev, Ew = np.linalg.eig(matrix)

    if all(Ev > 0):
        return "Positive definite"
    if all(Ev >= 0):
        return "Positve semi-definite"
    if all(Ev < 0):
        return "Negative definite"
    if all(Ev <= 0):
        return "Negative semi-definite"
    else:
        return 'Indefinite'
