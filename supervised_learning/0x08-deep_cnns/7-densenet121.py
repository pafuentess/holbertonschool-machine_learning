#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ doc """
    inputs = K.Input(shape=(224, 224, 3))

    x = K.layers.BatchNormalization()(inputs)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(64, kernel_size=(7, 7), padding='same',
                        kernel_initializer='he_normal',
                        strides=(2, 2))(x)
    x = K.layers.MaxPool2D((3, 3), (2, 2), padding='same')(x)

    x, flts = dense_block(x, 64, growth_rate, 6)
    x, flts = transition_layer(x, flts, compression)

    x, flts = dense_block(x, flts, growth_rate, 12)
    x, flts = transition_layer(x, flts, compression)

    x, flts = dense_block(x, flts, growth_rate, 24)
    x, flts = transition_layer(x, flts, compression)

    x, flts = dense_block(x, flts, growth_rate, 16)

    x = K.layers.AvgPool2D((7, 7), padding='same')(x)
    x = K.layers.Dense(1000, activation='softmax')(x)

    return K.Model(inputs, x)
