#!/usr/bin/env python3
""" doc """

from tensorflow import keras


def one_hot(labels, classes=None):
    """ doc """
    return keras.utils.to_categorical(labels)
