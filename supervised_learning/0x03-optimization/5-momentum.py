#!/usr/bin/env python3
""" doc """

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ doc """
    v = np.multiply(beta1, v) + np.multiply((1 - beta1), grad)
    var = var - np.multiply(alpha, v)

    return var, v