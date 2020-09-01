#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ doc """
    network.save_weights(filename)
    return None


def load_weights(network, filename):
    """ doc """
    network.load_weights(filename)
    return None
