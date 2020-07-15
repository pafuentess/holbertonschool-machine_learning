#!/usr/bin/env python3
""" doc """


def matrix_shape(matrix):
    """ doc """
    size = []
    while type(matrix) == list:
        size.append(len(matrix))
        matrix = matrix[0]
    return (size)


def add_matrices2D(mat1, mat2):
    """ doc """
    if len(arr1) != len(arr2):
        return (None)
    else:
        shape = matrix_shape(mat1)
        add_array = []
        for i in range(0, shape[0]):
            new_row = []
            for j in range(0, shape[1]):
                new_row.append(mat1[i][j] + mat2[i][j])
            add_array.append(new_row)
        return (add_array)
