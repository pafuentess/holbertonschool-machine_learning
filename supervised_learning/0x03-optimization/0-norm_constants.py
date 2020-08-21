#!/usr/bin/env python3
""" doc """

import numpy as np


def normalization_constants(X):
    """ doc """
    mean = np.mean(X, axis=0)
    dev = np.std(X, axis=0)

    return (mean, dev)
