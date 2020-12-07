#!/usr/bin/env python3
""" doc """

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ doc """

    t, m, i = X.shape

    H_f = []
    H_b = []
    h_f = h_0
    h_b = h_t

    H_f.append(h_0)
    H_b.append(h_t)

    for step in range(t):
        h_f = bi_cell.forward(h_f, X[step])
        h_b = bi_cell.backward(h_b, X[t - 1 - step])

        H_f.append(h_f)
        H_b.append(h_b)

    H_f = np.array(H_f)
    H_b = [x for x in reversed(H_b)]
    H_b = np.array(H_b)
    H = np.concatenate((H_f[1:], H_b[:-1]), axis=-1)

    Y = bi_cell.output(H)

    return H, Y
