#!/usr/bin/env python3
""" doc """

import numpy as np


class RNNCell:
    """ doc """
    def __init__(self, i, h, o):
        """ doc """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ doc """

        h_x = np.concatenate((h_prev, x_t), axis=1)
        f1 = np.matmul(h_x, self.Wh) + self.bh
        h_next = np.tanh(f1)

        f2 = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(f2) / np.sum(np.exp(f2), axis=1, keepdims=True)

        return h_next, y
