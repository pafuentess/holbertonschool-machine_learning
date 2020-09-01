#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K


def save_config(network, filename):
    """ doc """
    JsonModel = network.to_json()
    with open(filename, 'w') as f:
        f.write(JsonModel)
    return None


def load_config(filename):
    """ doc """
    with open(filename, 'r') as f:
        JsonModel = K.models.model_from_json(f.read())
    return JsonModel
