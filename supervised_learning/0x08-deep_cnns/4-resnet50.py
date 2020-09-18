#!/usr/bin/env python3
""" doc """
import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ doc """

    initialize = K.initializers.he_normal(seed=None)
    inputs = K.Input(shape=(224, 224, 3))
    x = K.layers.Conv2D(64, kernel_size=(7, 7),
                        padding='same', strides=(2, 2),
                        kernel_initializer=initialize)(inputs)

    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPool2D((3, 3), (2, 2), padding='same')(x)

    x = projection_block(x, [64, 64, 256], 1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    x = projection_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    x = projection_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    x = projection_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    x = K.layers.AvgPool2D((7, 7), padding='same')(x)
    x = K.layers.Dense(1000, activation='softmax')(x)

    return (K.Model(inputs, x))
