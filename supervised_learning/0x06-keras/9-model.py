#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K


def save_model(network, filename):
    """ doc """
    network.save(filename)
    return None


def load_model(filename):
    """ doc """
    load_file = K.models.load_model(filename)
    return load_file
