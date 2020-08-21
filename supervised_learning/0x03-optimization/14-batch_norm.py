#!/usr/bin/env python3
""" doc """

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ doc """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=None, kernel_initializer=init)

    z = layer(prev)

    m, s = tf.nn.moments(z, axes=[0])
    beta = tf.Variable(tf.zeros([z.get_shape()[-1]]))
    gamma = tf.Variable(tf.ones([z.get_shape()[-1]]))
    Znorm = tf.nn.batch_normalization(z, m, s, beta, gamma, 1e-8)

    return (activation(Znorm))
