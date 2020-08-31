#!/usr/bin/env python3
""" doc """

from tensorflow import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ doc """

    model = keras.Sequential()
    regulize = keras.regularizers.l2(lambtha)
    model.add(keras.layers.Dense(layers[0], activation=activations[0],
                                 input_shape=(nx,),
                                 kernel_regularizer=regulize,
                                 name='dense'))

    for i in range(1, len(layers)):
        model.add(keras.layers.Dropout(1 - keep_prob))
        model.add(keras.layers.Dense(layers[i], activation=activations[i],
                  kernel_regularizer=regulize, name=('dense_' + str(i))))

    return (model)
