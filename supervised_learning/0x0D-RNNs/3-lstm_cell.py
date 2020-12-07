#!/usr/bin/env python3
""" doc """

import numpy as np


class LSTMCell:
    """ doc """
    def __init__(self, i, h, o):
        """ doc """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """ doc """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """ doc """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """ doc """
        h_x = np.concatenate((h_prev, x_t), axis=1)

        f = self.sigmoid(np.matmul(h_x, self.Wf) + self.bf)
        u = self.sigmoid(np.matmul(h_x, self.Wu) + self.bu)
        Cg = np.tanh(np.matmul(h_x, self.Wc) + self.bc)

        Cnext = f * c_prev + u * Cg

        o = self.sigmoid(np. matmul(h_x, self.Wo) + self.bo)
        hn = o * np.tanh(Cnext)

        y = self.softmax(np.matmul(hn, self.Wy) + self.by)

        return hn, Cnext, y
