#!/usr/bin/env python3


def matrix_shape(matrix):
    size = []
    while type(matrix) == list:
        size.append(len(matrix))
        matrix = matrix[0]
    return (size)


def mat_mul(mat1, mat2):
    shape_mat1 = matrix_shape(mat1)
    shape_mat2 = matrix_shape(mat2)
    if (shape_mat1[1] != shape_mat2[0]):
        return (None)
    else:
        mul_matrix = []
        for row in range(0, len(mat1)):
            new_row = []
            for i in range(0, len(mat2[0])):
                sum = 0
                for j in range(0, len(mat1[0])):
                    mul = (mat1[row][j] * mat2[j][i])
                    sum += mul
                new_row.append(sum)
            mul_matrix.append(new_row)
        return mul_matrix
