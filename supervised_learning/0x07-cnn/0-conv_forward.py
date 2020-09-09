#!/usr/bin/env python3
""" doc """

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ doc """
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    Kh = W.shape[0]
    Kw = W.shape[1]

    if (type(padding) == tuple):
        Ph = padding[0]
        Pw = padding[1]

    if (padding == 'same'):
        Pw = int(((w_prev - 1) * stride[1] + Kw - w_prev) / 2) + 1
        Ph = int(((h_prev - 1) * stride[0] + Kh - h_prev) / 2) + 1

    if (padding == 'valid'):
        Ph = 0
        Pw = 0

    A_prev = np.pad(A_prev, [(0, 0), (Ph, Ph), (Pw, Pw), (0, 0)],
                    constant_values=0)

    convolutedH = int(((h_prev - Kh + (2 * Ph)) / stride[0]) + 1)
    convolutedW = int(((w_prev - Kw + (2 * Pw)) / stride[1]) + 1)

    convolutedImage = np.zeros((m, convolutedH,
                                convolutedW, W.shape[3]))
    Nlayer = np.arange(0, m)

    for i in range(convolutedH):
        for j in range(convolutedW):
            for k in range(W.shape[3]):              
                val = np.sum(np.multiply(A_prev[Nlayer,
                                         i*stride[0]:Kh + (i*stride[0]),
                                         j*stride[1]:Kw + (j*stride[1])],
                                         W[:, :, :, k]), axis=(1, 2, 3))
                bias = b[:, :, :, k]
                convolutedImage[Nlayer, i, j, k] = activation((val + bias))

    return(convolutedImage)
