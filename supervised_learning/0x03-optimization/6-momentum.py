#!/usr/bin/env python3
""" doc """
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ doc """
    optimize = tf.train.MomentumOptimizer(alpha, beta1)
    train = optimize.minimize(loss)
    return train
