#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ doc """
    initialize = K.initializers.he_normal(seed=None)

    layer1 = K.layers.Conv2D(filters=filters[0], kernel_size=1,
                             padding='same', activation='relu',
                             kernel_initializer=initialize)(A_prev)

    layer2r = K.layers.Conv2D(filters=filters[0], kernel_size=1,
                              padding='same', activation='relu',
                              kernel_initializer=initialize)(A_prev)

    layer2 = K.layers.Conv2D(filters=filters[2], kernel_size=1,
                             padding='same', activation='relu',
                             kernel_initializer=initialize)(layer2r)

    layer3r = K.layers.Conv2D(filters=filters[3], kernel_size=1,
                              padding='same', activation='relu',
                              kernel_initializer=initialize)(A_prev)

    layer3 = K.layers.Conv2D(filters=filters[4], kernel_size=1,
                             padding='same', activation='relu',
                             kernel_initializer=initialize)(layer3r)

    poolLayer = K.layers.MaxPooling2D(pool_size=[3, 3], strides=1,
                                      padding='same')(A_prev)

    poolLayerR = K.layers.Conv2D(filters=filters[5], kernel_size=1,
                                 padding='same', activation='relu',
                                 kernel_initializer=initialize)(poolLayer)

    layer_list = [layer1, layer2, layer3, poolLayerR]

    return (K.layers.concatenate(layer_list))
