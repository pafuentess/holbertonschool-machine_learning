#!/usr/bin/env python3
""" doc """

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ doc """
    train = tf.train.inverse_time_decay(alpha, decay_rate=decay_rate,
                                        global_step=global_step,
                                        decay_steps=decay_step,
                                        staircase=True)
    return train
