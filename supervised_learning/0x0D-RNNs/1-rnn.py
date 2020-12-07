#!/usr/bin/env python3
""" doc """

import numpy as np


def rnn(rnn_cell, X, h_0):
    """ doc """
    t = X.shape[0]

    hidden = []
    output = []

    hidden.append(h_0)
    for step in range(t):
        h, y = rnn_cell.forward(hidden[-1], X[step])
        hidden.append(h)
        output.append(y)

    return np.array(hidden), np.array(output)
