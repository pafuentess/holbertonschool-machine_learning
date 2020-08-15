#!/usr/bin/env python3
""" doc """

import tensorflow as tf


def create_train_op(loss, alpha):
    """ doc """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
