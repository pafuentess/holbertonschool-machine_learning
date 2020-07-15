#!/usr/bin/env python3
""" doc """


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ doc """
    return (np.concatenate((mat1, mat2), axis))
