#!/usr/bin/env python3
""" doc """


def matrix_shape(matrix):
    """ doc """
    size = []
    while type(matrix) == list:
        size.append(len(matrix))
        matrix = matrix[0]
    return (size)


def add_arrays(arr1, arr2):
    """ doc """
    if (matrix_shape(arr1) != (matrix_shape(arr2))):
        return (None)
    else:
        shape = matrix_shape(arr1)
        new_array = []
        for i in range(0, shape[0]):
            new_array.append(arr1[i] + arr2[i])
        return (new_array)
