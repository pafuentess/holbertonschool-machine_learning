#!/usr/bin/env python3
""" doc """


def np_elementwise(mat1, mat2):
    """ doc """
    sum_matrix = mat1 + mat2
    mul_matrix = mat1 * mat2
    diference_matrix = mat1 - mat2
    div_matrix = mat1 / mat2

    operators = (sum_matrix, diference_matrix, mul_matrix, div_matrix)
    return (operators)
