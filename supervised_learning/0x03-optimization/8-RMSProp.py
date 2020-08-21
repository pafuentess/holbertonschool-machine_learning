#!/usr/bin/env python3
""" doc """

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ doc """
    optimize = tf.train.RMSPropOptimizer(alpha, epsilon=epsilon, decay=beta2)
    train = optimize.minimize(loss)
    return train
