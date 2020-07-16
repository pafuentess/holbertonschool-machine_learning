#!/usr/bin/env python3
""" doc """

import numpy as np


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
    return (new_mat[slice_list])
