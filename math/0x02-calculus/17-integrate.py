#!/usr/bin/env python3
""" doc """


def poly_integral(poly, C=0):
    """ doc """
    if type(C) is int or type(C) is float:
        if len(poly) == 1 and poly[0] == 0:
            return C

        new_poly = [C]
        for i in range(len(poly)):
            if type(poly[i]) is int or type(poly[i]) is float:
                pass
            else:
                return None
        for i in range(0, len(poly)):
            operation = poly[i] / (i + 1)
            if operation.is_integer():
                operation = int(operation)
            new_poly.append(operation)
        return (new_poly)
    else:
        return None
