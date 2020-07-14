#!/usr/bin/env python3


def matrix_shape(matrix):
    size = []
    while type(matrix) == list:
        size.append(len(matrix))
        matrix = matrix[0]
    return (size)


def matrix_transpose(matrix):
    shape = matrix_shape(matrix)
    matrix_T = []
    if (shape[0] == shape[1]):
        for i in range(0, shape[0]):
            new_array = []
            for j in range(0, shape[1]):
                new_array.append(matrix[j][i])
            matrix_T.append(new_array)
        return (matrix_T)
    else:
        for i in range(0, shape[1]):
            new_array = []
            for j in range(0, shape[0]):
                new_array.append(matrix[j][i])
            matrix_T.append(new_array)
        return (matrix_T)
