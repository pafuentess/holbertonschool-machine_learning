#!/usr/bin/env python3
""" doc """

import tensorflow as tf


def lenet5(x, y):
    """ doc """

    initializer = tf.contrib.layers.variance_scaling_initializer()

    C_layer1 = tf.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                                activation=tf.nn.relu,
                                kernel_initializer=initializer)(x)

    P_layer1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(C_layer1)

    C_layer2 = tf.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                                activation=tf.nn.relu,
                                kernel_initializer=initializer)(P_layer1)

    P_layer2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(C_layer2)

    flatten = tf.layers.Flatten()(P_layer2)

    F_layer1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                               kernel_initializer=initializer)(flatten)

    F_layer2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                               kernel_initializer=initializer)(F_layer1)

    L_output = tf.layers.Dense(units=10,
                               kernel_initializer=initializer)(F_layer2)

    Y = tf.nn.softmax(L_output)

    predicted = tf.argmax(y, 1)
    opteined = tf.argmax(L_output, 1)
    equal = tf.equal(predicted, opteined)
    accuar = tf.reduce_mean(tf.cast(equal, tf.float32))

    loss = tf.losses.softmax_cross_entropy(Y, L_output)

    optimize = tf.train.AdamOptimizer().minimize(loss)

    return Y, optimize, loss, accuar
