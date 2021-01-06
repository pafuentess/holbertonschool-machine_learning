#!/usr/bin/env python3
""" doc """

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ doc """
    def __init__(self, units):
        """ doc """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """ doc """
        s_prev_expand = tf.expand_dims(s_prev, axis=1)
        s = self.V(tf.nn.tanh(self.W(s_prev_expand) + self.U(hidden_states)))
        W = tf.nn.softmax(s, axis=1)
        context = tf.reduce_sum(W * hidden_states, axis=1)

        return context, W
