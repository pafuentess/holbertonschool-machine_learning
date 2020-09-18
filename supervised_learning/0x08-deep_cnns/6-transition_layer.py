#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ doc """
    x = K.layers.BatchNormalization()(X)
    x = K.layers.Activation('relu')(x)
    n = int(nb_filters * compression)
    x = K.layers.Conv2D(n, kernel_size=(1, 1),
                        padding='same', kernel_initializer='he_normal')(x)
    x = K.layers.AvgPool2D((2, 2), padding='same')(x)
    return (x, n)
