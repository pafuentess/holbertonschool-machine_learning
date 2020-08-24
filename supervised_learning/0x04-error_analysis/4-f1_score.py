#!/usr/bin/env python3
""" doc """

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ doc """
    Sensitivity = sensitivity(confusion)
    Precision = precision(confusion)
    F1 = 2 * ((Sensitivity * Precision) / (Sensitivity + Precision))
    return F1
