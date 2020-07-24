#!/usr/bin/env python3
""" doc """


def check_0(new_poly):
    """ doc """
    for i in range(len(new_poly) - 1, 0, -1):
        if new_poly[i] == 0:
            new_poly.pop()
        else:
            break
    return new_poly


def poly_integral(poly, C=0):
    """ doc """

    if poly is None or poly == [] or type(poly) is not list:
        return None

    if type(C) is int or type(C) is float:
        if poly == [0]:
            return C
        if C % 1 == 0:
            C = int(C)

        new_poly = [C]
        for i in range(0, len(poly)):
            if type(poly[i]) is int or type(poly[i]) is float:
                operation = poly[i] / (i + 1)
                if operation.is_integer():
                    operation = int(operation)
                new_poly.append(operation)
            else:
                return None
        check = check_0(new_poly)
        return (check)
    else:
        return None
