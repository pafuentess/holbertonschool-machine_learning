#!/usr/bin/env python3
""" doc """

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ doc """

    layers = images.shape[0]
    LayerH = images.shape[1]
    layerW = images.shape[2]

    kernelH = kernel.shape[0]
    kernelW = kernel.shape[1]

    convolutedW = layerW - kernelW + 1
    convolutedH = LayerH - kernelH + 1

    convolutedImage = np.zeros((layers, convolutedH, convolutedW))
    Nlayer = np.arange(0, layers)

    for i in range(convolutedH):
        for j in range(convolutedW):
            val = np.sum(np.multiply(images[Nlayer, i:kernelH + i,
                         j:kernelW + j], kernel), axis=(1, 2))
            convolutedImage[Nlayer, i, j] = val
    return(convolutedImage)
