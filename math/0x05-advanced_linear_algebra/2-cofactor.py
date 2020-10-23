#!/usr/bin/env python3
""" doc """


def cofactor(matrix):
    """ doc """
    Minor = minor(matrix)
    Cofactor = []

    for i in range(len(matrix)):
        Cofactor.append([])
        for j in range(len(matrix)):
            coef = (((-1) ** (i + j)) * Minor[i][j])
            Cofactor[i].append(coef)
    return Cofactor


def minor(matrix):
    """ doc """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')

    for element in matrix:
        if not isinstance(element, list):
            raise TypeError('matrix must be a list of lists')

    my_len = len(matrix)
    if my_len == 1 and len(matrix[0]) == 0:
        raise ValueError('matrix must be a non-empty square matrix')

    for element in matrix:
        if len(element) != my_len:
            raise ValueError('matrix must be a non-empty square matrix')

    if my_len == 1:
        return [[1]]

    minor = []
    for i in range(my_len):
        minor.append([])
        for j in range(my_len):
            rows = [matrix[m] for m in range(my_len) if m != i]
            new_m = [[row[n] for n in range(my_len) if n != j] for row in rows]
            my_det = determinant(new_m)
            minor[i].append(my_det)

    return minor


def mminor(matrix):
    """ doc """
    coef1 = matrix[0][0] * matrix[1][1]
    coef2 = matrix[0][1] * matrix[1][0]
    return coef1 - coef2


def determinant(matrix):
    """ doc """

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of list")

    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return (mminor(matrix))

    det = 0
    x = len(matrix)
    for i, k in enumerate(matrix[0]):
        rows = [row for row in matrix[1:]]
        new_m = [[row[n] for n in range(x) if n != i] for row in rows]
        det += k * (-1) ** i * determinant(new_m)

    return det
