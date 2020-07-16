#!/usr/bin/env python3
""" doc """


def add_arrays(arr1, arr2):
    """ doc """
    if len(arr1) != len(arr2):
        return (None)

    new_array = []
    for i in range(len(arr1)):
        new_array.append(arr1[i] + arr2[i])
    return (new_array)


def add_matrices2D(mat1, mat2):
    """ doc """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return (None)

    add_array = []
    for i in range(len(mat1)):
        new_row = []
        for j in range(len(mat1[i])):
            new_row.append(mat1[i][j] + mat2[i][j])
        add_array.append(new_row)
    return (add_array)


def matrix_shape(matrix):
    """ doc """
    size = []
    while type(matrix) == list:
        size.append(len(matrix))
        matrix = matrix[0]
    return (size)


def add_matrices(mat1, mat2):
    """ doc """
    if (matrix_shape(mat1) != matrix_shape(mat2)):
        return None
    if (len(mat1) == len(mat2) and (type(mat1[0]) is int)):
        return(add_arrays(mat1, mat2))
    if (len(mat1) == len(mat2) and type(mat1[0] == list)):
        if type(mat1[0][0]) is int:
            return (add_matrices2D(mat1, mat2))
    result = []
    for i in range(len(mat1)):
        result.append(add_matrices(mat1[i], mat2[i]))
    return (result)
