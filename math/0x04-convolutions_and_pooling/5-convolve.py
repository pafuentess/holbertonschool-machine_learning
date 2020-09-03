#!/usr/bin/env python3
""" doc """

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ doc """

    layers = images.shape[0]
    layerH = images.shape[1]
    layerW = images.shape[2]

    kernelH = kernels.shape[0]
    kernelW = kernels.shape[1]

    if (type(padding) == tuple):
        Ph = padding[0]
        Pw = padding[1]

    if (padding == 'same'):
        Pw = int(((layerW - 1) * stride[1] + kernelW - layerW) / 2) + 1
        Ph = int(((layerH - 1) * stride[0] + kernelH - layerH) / 2) + 1

    if (padding == 'valid'):
        Ph = 0
        Pw = 0

    images = np.pad(images, [(0, 0), (Ph, Ph), (Pw, Pw), (0, 0)],
                    constant_values=0)

    convolutedH = int(((layerH - kernelH + (2 * Ph)) / stride[0]) + 1)
    convolutedW = int(((layerW - kernelW + (2 * Pw)) / stride[1]) + 1)

    convolutedImage = np.zeros((layers, convolutedH,
                                convolutedW, kernels.shape[3]))
    Nlayer = np.arange(0, layers)

    for i in range(convolutedH):
        for j in range(convolutedW):
            for k in range(kernels.shape[3]):
                val = np.sum(np.multiply(images[Nlayer,
                                         i*stride[0]:kernelH + (i*stride[0]),
                                         j*stride[1]:kernelW + (j*stride[1])],
                                         kernels[:, :, :, k]), axis=(1, 2, 3))

                convolutedImage[Nlayer, i, j, k] = val

    return(convolutedImage)
