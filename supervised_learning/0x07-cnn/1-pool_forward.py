#!/usr/bin/env python3
""" doc """

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ doc """

    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kernelH = kernel_shape[0]
    kernelW = kernel_shape[1]

    convolutedH = int(((h_prev - kernelH) / stride[0]) + 1)
    convolutedW = int(((w_prev - kernelW) / stride[1]) + 1)

    convolutedImage = np.zeros((m, convolutedH,
                                convolutedW,
                                A_prev.shape[3]))
    Nlayer = np.arange(0, m)

    for i in range(convolutedH):
        for j in range(convolutedW):
            if mode == 'avg':
                val = np.mean(A_prev[Nlayer,
                              i*stride[0]:kernelH + (i*stride[0]),
                              j*stride[1]:kernelW + (j*stride[1])],
                              axis=(1, 2))
            if mode == 'max':
                val = np.max(A_prev[Nlayer,
                             i*stride[0]:kernelH + (i*stride[0]),
                             j*stride[1]:kernelW + (j*stride[1])],
                             axis=(1, 2))

            convolutedImage[Nlayer, i, j] = val

    return(convolutedImage)
