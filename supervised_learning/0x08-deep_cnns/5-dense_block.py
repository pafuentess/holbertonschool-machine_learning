#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ doc """

    for i in range(layers):
        x = K.layers.BatchNormalization()(X)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(growth_rate * 4, kernel_size=(1, 1),
                            padding='same',  strides=(1, 1),
                            kernel_initializer='he_normal')(x)
        x = K.layers.BatchNormalization()(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(growth_rate, kernel_size=(3, 3),
                            padding='same',  strides=(1, 1),
                            kernel_initializer='he_normal')(x)
        X = K.layers.concatenate([X, x])
        nb_filters += growth_rate

    return (X, nb_filters)
