#!/usr/bin/env python3
""" doc """

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ doc """
    t, m, i = X.shape
    hs = h_0.shape[2]
    cells = len(rnn_cells)

    H = np.zeros((t + 1, cells, m, hs))
    H[0] = h_0

    for step in range(t):
        for layer in range(cells):
            if layer == 0:
                h, y = rnn_cells[layer].forward(H[step, layer], X[step])
            else:
                h, y = rnn_cells[layer].forward(H[step, layer], h)

            H[step + 1, layer, ...] = h

            if layer == cells - 1:
                if step == 0:
                    Y = y
                else:
                    Y = np.concatenate((Y, y))

    Y = Y.reshape(t, m, Y.shape[-1])

    return H, Y
