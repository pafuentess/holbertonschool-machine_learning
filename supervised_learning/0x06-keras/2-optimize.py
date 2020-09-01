#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ doc """
    opt = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(loss='categorical_crossentropy',
                    optimizer=opt, metrics=['accuracy'])
    return(None)
