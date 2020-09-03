#!/usr/bin/env python3
""" doc """

import numpy as np


def convolve_grayscale_same(images, kernel):
    """ doc """

    layers = images.shape[0]
    layerH = images.shape[1]
    layerW = images.shape[2]

    kernelH = kernel.shape[0]
    kernelW = kernel.shape[1]

    Ph = int((kernelH) / 2)
    Pw = int((kernelW) / 2)

    images = np.pad(images, [(0, 0), (Ph, Ph), (Pw, Pw)],
                    'constant', constant_values=0)

    convolutedImage = np.zeros((layers, layerH, layerW))
    Nlayer = np.arange(0, layers)

    for i in range(layerH):
        for j in range(layerW):
            val = np.sum(np.multiply(images[Nlayer, i:kernelH + i,
                         j:kernelW + j], kernel), axis=(1, 2))
            convolutedImage[Nlayer, i, j] = val

    return(convolutedImage)
