#!/usr/bin/env python3
""" doc """

import tensorflow as tf


def create_layer(prev, n, activation):
    """ doc """
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=initialize,
                            activation=activation, name="layer")
    return (layer(prev))
