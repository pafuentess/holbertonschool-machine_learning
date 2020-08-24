#!/usr/bin/env python3
""" doc """

import numpy as np


def specificity(confusion):
    """ doc """
    Tp = np.diagonal(confusion)
    Fn = np.sum(confusion, axis=1) - Tp
    Fp = np.sum(confusion, axis=0) - Tp
    Tn = np.sum(confusion) - (Tp + Fn + Fp)
    Specificity = Tn / (Fp + Tn)
    return Specificity
