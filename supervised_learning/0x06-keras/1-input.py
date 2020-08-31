#!/usr/bin/env python3
""" doc """

from tensorflow import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ doc """

    inputs = keras.Input(shape=(nx,))
    regulize = keras.regularizers.l2(lambtha)
    output = keras.layers.Dense(layers[0], activation=activations[0],
                                input_shape=(nx,),
                                kernel_regularizer=regulize,
                                name='dense')(inputs)

    for i in range(1, len(layers)):
        drop = (keras.layers.Dropout(1 - keep_prob))(output)
        output = keras.layers.Dense(layers[i], activation=activations[i],
                                    kernel_regularizer=regulize,
                                    name=('dense_' + str(i)))(drop)

    model = keras.Model(inputs=inputs, outputs=output)
    return (model)
