#!/usr/bin/env python3
""" doc """

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ doc """
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    regulize = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, kernel_initializer=initialize,
                            kernel_regularizer=regulize,
                            activation=activation, name="layer")
    return layer(prev)
