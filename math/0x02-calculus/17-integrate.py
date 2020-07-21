#!/usr/bin/env python3
""" doc """


def poly_integral(poly, C=0):
    """ doc """
    if type(C) is int or type(C) is float:
        if len(poly) == 1 and poly[0] == 0:
            return C

        new_poly = []
        for i in range(len(poly)):
            if type(poly[i]) is int or type(poly[i]) is float:
                new_poly.append(0)
            else:
                return None
        new_poly.append(0)

        for i in range(0, len(poly)):
            operation = poly[i] / (i + 1)
            if operation.is_integer():
                operation = int(operation)
            new_poly[i + 1] = operation
        return (new_poly)
    else:
        return None
