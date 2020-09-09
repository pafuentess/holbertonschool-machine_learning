#!/usr/bin/env python3
""" doc """

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ doc """

    m = dA.shape[0]
    h_new = dA.shape[1]
    w_new = dA.shape[2]
    c_new = dA.shape[3]

    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    Kh = kernel_shape[0]
    Kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    dx = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    if mode == 'max':
                        A_aux = A_prev[i, h*sh:Kh+(h*sh), w*sw:Kw+(w*sw), c]
                        mask = (A_aux == np.max(A_aux))
                        dx[i, h*sh:Kh+(h*sh), w*sw:Kw+(w*sw), c] += dA[i,
                                                                       h,
                                                                       w,
                                                                       c]*mask

                    if mode == 'avg':
                        dx[i, h*sh:Kh+(h*sh), w*sw:Kw+(w*sw), c] += dA[i,
                                                                       h,
                                                                       w,
                                                                       c]/Kh/Kw
    return dx
