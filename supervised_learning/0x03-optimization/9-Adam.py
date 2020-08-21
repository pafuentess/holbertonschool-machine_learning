#!/usr/bin/env python3
""" doc"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ doc """

    v = v * beta1 + ((1 - beta1) * grad)
    s = s * beta2 + ((1 - beta2) * (grad ** 2))
    Vcorrect = v / (1 - (beta1 ** t))
    Scorect = s / (1 - (beta2 ** t))
    var = var - (alpha * (Vcorrect / ((Scorect ** 0.5) + epsilon)))
    return var, v, s
