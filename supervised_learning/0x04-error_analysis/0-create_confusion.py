#!/usr/bin/env python3
""" doc """

import numpy as np


def create_confusion_matrix(labels, logits):
    """ doc """
    return np.matmul(labels.T, logits)
