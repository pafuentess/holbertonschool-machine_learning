#!/usr/bin/env python3
""" doc """


def poly_derivative(poly):
    """ doc """
    if poly is None or poly == [] or type(poly) is not list:
        return None
    new_poly = []
    if len(poly) == 1:
        return [0]
    for i in range(1, len(poly)):
        if type(poly[i]) is str:
            return None
        new_poly.append(poly[i] * i)
    return (new_poly)
