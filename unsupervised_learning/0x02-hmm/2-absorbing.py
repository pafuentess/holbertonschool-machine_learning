#!/usr/bin/env python3
""" doc """
import numpy as np


def absorbing(P):
    """ doc """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    absState = np.where(np.diag(P) == 1)
    row = P[absState[0]]
    account = np.sum(row, axis=0)
    for i in range(P.shape[0]):
        check_row = P[i] != 0
        intersection = account * check_row
        if (intersection == 1).any():
            account[i] = 1
    return account.all()
