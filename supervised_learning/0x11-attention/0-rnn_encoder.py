#!/usr/bin/env python3
""" doc """

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ doc """
    def __init__(self, vocab, embedding, units, batch):
        """ doc """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """ doc """
        init = tf.keras.initializers.Zeros()
        return (init(shape=(self.batch, self.units)))

    def call(self, x, initial):
        """ doc """
        embedding = self.embedding(x)
        outputs, hidden = self.gru(embedding, initial_state=initial)
        return outputs, hidden
