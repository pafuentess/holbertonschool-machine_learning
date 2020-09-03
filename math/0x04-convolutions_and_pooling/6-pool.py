#!/usr/bin/env python3
""" doc """

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """ doc """

    layers = images.shape[0]
    layerH = images.shape[1]
    layerW = images.shape[2]

    kernelH = kernel_shape[0]
    kernelW = kernel_shape[1]

    convolutedH = int(((layerH - kernelH) / stride[0]) + 1)
    convolutedW = int(((layerW - kernelW) / stride[1]) + 1)

    convolutedImage = np.zeros((layers, convolutedH,
                                convolutedW,
                                images.shape[3]))
    Nlayer = np.arange(0, layers)

    for i in range(convolutedH):
        for j in range(convolutedW):
            if mode == 'avg':
                val = np.mean(images[Nlayer,
                              i*stride[0]:kernelH + (i*stride[0]),
                              j*stride[1]:kernelW + (j*stride[1])],
                              axis=(1, 2))
            if mode == 'max':
                val = np.max(images[Nlayer,
                             i*stride[0]:kernelH + (i*stride[0]),
                             j*stride[1]:kernelW + (j*stride[1])],
                             axis=(1, 2))

            convolutedImage[Nlayer, i, j] = val

    return(convolutedImage)
