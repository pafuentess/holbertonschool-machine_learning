#!/usr/bin/env python3
""" doc """
import numpy as np


def sensitivity(confusion):
    """ doc """
    Tp = np.diagonal(confusion)
    Fn = np.sum(confusion, axis=1)
    Sentivity = Tp / Fn
    return Sentivity
