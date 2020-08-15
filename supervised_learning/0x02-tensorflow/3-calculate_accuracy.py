#!/usr/bin/env python3
""" doc """

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ doc """
    y_max = tf.argmax(y, axis=1)
    y_pred_max = tf.argmax(y_pred, axis=1)
    equal = tf.equal(y_max, y_pred_max)
    cast = tf.cast(equal, tf.float32)
    accuar = tf.reduce_mean(cast)
    return (accuar)
