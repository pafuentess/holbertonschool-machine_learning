#!/usr/bin/env python3

import numpy as np


class BidirectionalCell:
    """ doc """
    def __init__(self, i, h, o):
        """ doc """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(i + h + o, o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ doc """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        Z_next = np.matmul(h_x, self.Whf) + self.bhf
        return np.tanh(Z_next)

    def backward(self, h_next, x_t):
        """ doc """
        h_x = np.concatenate((h_next, x_t), axis=1)
        Z_prev = np.matmul(h_x, self.Whb) + self.bhb
        h_prev = np.tanh(Z_prev)

        return h_prev

    def softmax(self, x):
        """ doc """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def output(self, H):
        """ doc """
        t = H.shape[0]

        Y = []

        for step in range(t):
            # input for softmax activation
            Z_y = np.matmul(H[step], self.Wy) + self.by

            # softmax activation
            y = self.softmax(Z_y)
            Y.append(y)

        Y = np.array(Y)

        return Y
