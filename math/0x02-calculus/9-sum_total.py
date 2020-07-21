#!/usr/bin/env python3
""" doc """


def summation_i_squared(n):
    """ doc """
    if type(n) is not int:
        return (None)
    result = int((n * (n + 1) * ((2 * n) + 1)) / 6)
    return (result)
