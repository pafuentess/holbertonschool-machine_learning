#!/usr/bin/env python3
""" doc """


def cat_matrices2D(mat1, mat2, axis=0):
    """ doc """
    if (len(mat1) != len(mat2) and axis == 1):
        return (None)
    if (len(mat1[0]) != len(mat2[0]) and axis == 0):
        return (None)
    mat_concatenated = []
    new_mat1 = mat1[:]
    if (axis == 0):
        mat_concatenated = new_mat1 + mat2
        return (mat_concatenated)
    elif (axis == 1):
        j = 0
        for i in new_mat1:
            mat_concatenated.append(i + mat2[j])
            j += 1
        return mat_concatenated
