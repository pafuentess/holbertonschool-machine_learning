#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ doc """

    initialize = K.initializers.he_normal(seed=None)

    layer1 = K.layers.Conv2D(filters[0], kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=initialize)(A_prev)
    batch = K.layers.BatchNormalization(axis=3)(layer1)
    activation = K.layers.Activation('relu')(batch)

    layer2 = K.layers.Conv2D(filters[1], kernel_size=(3, 3),
                             padding='same',
                             kernel_initializer=initialize)(activation)
    batch = K.layers.BatchNormalization(axis=3)(layer2)
    activation = K.layers.Activation('relu')(batch)

    layer3 = K.layers.Conv2D(filters[2], kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=initialize)(activation)
    batch = K.layers.BatchNormalization(axis=3)(layer3)

    allTogueter = K.layers.Add()([batch, A_prev])

    return K.layers.Activation('relu')(allTogueter)
