#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ doc """
    optimize = tf.train.AdamOptimizer(learning_rate=alpha,
                                      beta1=beta1, beta2=beta2,
                                      epsilon=epsilon)
    train = optimize.minimize(loss)
    return train
