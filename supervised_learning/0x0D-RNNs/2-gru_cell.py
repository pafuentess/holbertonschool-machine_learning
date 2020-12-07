#!/usr/bin/env python3
""" doc """

import numpy as np


class GRUCell:
    """ doc """
    def __init__(self, i, h, o):
        """ doc """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """ doc """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """ doc """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ doc """
        h_x = np.concatenate((h_prev, x_t), axis=1)

        z = self.sigmoid(np.matmul(h_x, self.Wz) + self.bz)
        r = self.sigmoid(np.matmul(h_x, self.Wr) + self.br)

        h_x = np.concatenate((r * h_prev, x_t), axis=1)

        Zn = np. matmul(h_x, self.Wh) + self.bh

        hc = np.tanh(Zn)

        hn = (1 - z) * h_prev + z * hc

        y = self.softmax(np.matmul(hn, self.Wy) + self.by)

        return hn, y
