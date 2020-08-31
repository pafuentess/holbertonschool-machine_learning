#!/usr/bin/env python3
""" doc """

from tensorflow import keras


def optimize_model(network, alpha, beta1, beta2):
    """ doc """
    opt = keras.optimizers.Adam(alpha, beta1, beta2)
    network.compile(loss='categorical_crossentropy',
                    optimizer=opt, metrics=['accuracy'])
    return(None)
