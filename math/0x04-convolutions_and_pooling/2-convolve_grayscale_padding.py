#!/usr/bin/env python3
""" doc """

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ doc """

    layers = images.shape[0]
    layerH = images.shape[1]
    layerW = images.shape[2]

    kernelH = kernel.shape[0]
    kernelW = kernel.shape[1]

    convolutedH = layerH - kernelH + (2 * padding[0]) + 1
    convolutedW = layerW - kernelW + (2 * padding[1]) + 1

    images = np.pad(images, [(0, 0), (padding[0], padding[0]), (padding[1],
                             padding[1])], constant_values=0)

    convolutedImage = np.zeros((layers, convolutedH, convolutedW))
    Nlayer = np.arange(0, layers)

    for i in range(convolutedH):
        for j in range(convolutedW):
            val = np.sum(np.multiply(images[Nlayer, i:kernelH + i,
                         j:kernelW + j], kernel), axis=(1, 2))
            convolutedImage[Nlayer, i, j] = val

    return(convolutedImage)
