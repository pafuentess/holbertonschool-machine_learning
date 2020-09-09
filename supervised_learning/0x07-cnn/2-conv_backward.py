#!/usr/bin/env python3
""" doc """

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ doc """

    m = dZ.shape[0]
    h_new = dZ.shape[1]
    w_new = dZ.shape[2]
    c_new = dZ.shape[3]

    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    Kh = W.shape[0]
    Kw = W.shape[1]

    sh = stride[0]
    sw = stride[1]

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

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    dx = np.zeros(A_prev.shape)
    dw = np.zeros(W.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    w_aux = W[:, :, :, c]
                    dz_aux = dZ[i, h, w, c]
                    dx[i, h*sh: h*sh+Kh, w*sw: w*sw+Kw, :] += dz_aux * w_aux
                    A_aux = A_prev[i, h*sh: h*sh+Kh, w*sw: w*sw+Kw, :]
                    dw[:, :, :, c] += dz_aux * A_aux

    dx = dx[:, Ph:dx.shape[1]-Ph, Pw:dx.shape[2]-Pw, :]

    return(dx, dw, db)
