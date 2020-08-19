#!/usr/bin/env python3
""" doc """

def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ doc """
    s = (beta2 * s) + ((1-beta2) * (grad**2))
    var = var - (alpha * grad * (1 / ((s**0.5) + epsilon)))
    return var, s