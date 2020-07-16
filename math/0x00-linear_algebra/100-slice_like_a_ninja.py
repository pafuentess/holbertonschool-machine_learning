#!/usr/bin/env python3
""" doc """


def np_slice(matrix, axes={}):
    """ doc """
    new_mat = matrix.copy()
    slice_list = []
    for i in range((max(axes.keys()) + 1)):
        values = axes.get(i)
        if values:
            slice_list.append(slice(*values))
        else:
            slice_list.append(slice(None))
    tuple_list = tuple(slice_list)
    return (new_mat[tuple_list])
