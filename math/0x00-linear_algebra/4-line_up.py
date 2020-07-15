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
    if len(arr1) != len(arr2):
        return (None)

    new_array = []
    for i in range(len(arr1)):
        new_array.append(arr1[i] + arr2[i])
    return (new_array)
