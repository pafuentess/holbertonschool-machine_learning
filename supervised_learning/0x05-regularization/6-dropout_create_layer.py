#!/usr/bin/env python3
""" doc """

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ doc """
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regulize = tf.layers.Dropout(keep_prob)
    layer = tf.layers.Dense(units=n, kernel_initializer=initialize,
                            kernel_regularizer=regulize,
                            activation=activation, name="layer")
    return layer(prev)
