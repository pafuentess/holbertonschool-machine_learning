#!/usr/bin/env python3


def matrix_shape(matrix):
    """ doc """
    size = []
    while type(matrix) == list:
        size.append(len(matrix))
        matrix = matrix[0]
    return (size)

