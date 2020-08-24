#!/usr/bin/env python3
""" doc """

import numpy as np


def precision(confusion):
    """ doc """
    Tp = np.diagonal(confusion)
    Fp = np.sum(confusion, axis=0)
    Precision = Tp / Fp
    return Precision
