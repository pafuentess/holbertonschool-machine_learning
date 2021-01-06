#!/usr/bin/env python3
""" doc """

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ doc """
    def __init__(self, vocab, embedding, units, batch):
        """ doc """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """ doc """
        batch, units = s_prev.shape
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        Embedding = self.embedding(x)
        context_expand = tf.expand_dims(context, axis=1)
        concatenateInput = tf.concat([context_expand, Embedding], axis=-1)

        outputs, hidden = self.gru(concatenateInput)
        outputs = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))

        Y = self.F(outputs)

        return (Y, hidden)
