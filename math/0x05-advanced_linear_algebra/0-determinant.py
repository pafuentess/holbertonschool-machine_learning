#!/usr/bin/env python3
""" doc """


def minor(matrix):
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
        return (minor(matrix))

    det = 0

    for i, j in enumerate(matrix[0]):
        row = [roww for roww in matrix[1:]]
        new_matrix = []
        for roww in row:
            line = []
            for col in range(len(matrix)):
                if col != 1:
                    line.append(roww[col])
            new_matrix.append(line)

        det = det + j * (-1) ** i * determinant(new_matrix)

    return det
