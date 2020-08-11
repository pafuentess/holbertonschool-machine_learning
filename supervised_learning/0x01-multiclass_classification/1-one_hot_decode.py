#!/usr/bin/env python3

import numpy as np


def one_hot_decode(one_hot):
    """ doc """
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot.shape) != 2:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    y_decode = np.argmax(one_hot, axis=0)
    return (y_decode)
