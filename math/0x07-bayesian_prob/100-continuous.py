#!/usr/bin/env python3
""" doc """

from scipy import special


def posterior(x, n, p1, p2):
    """ doc """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        m = "x must be an integer that is greater than or equal to 0"
        raise ValueError(m)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if (not isinstance(p1, float)) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if (not isinstance(p2, float)) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    f1 = x + 1
    f2 = n - x + 1

    acum1 = special.btdtr(f1, f2, p1)
    acum2 = special.btdtr(f1, f2, p2)

    return acum2 - acum1
