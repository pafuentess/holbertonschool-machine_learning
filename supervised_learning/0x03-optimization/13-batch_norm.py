#!/usr/bin/env python3
""" doc """

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ doc """
    mean = Z.mean(0)
    var = Z.var(0)
    Znor = (Z - mean) / (var + epsilon) ** (0.5)
    return((gamma * Znor) + beta)
