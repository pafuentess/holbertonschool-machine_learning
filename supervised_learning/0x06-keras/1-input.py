#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ doc """

    inputs = K.Input(shape=(nx,))
    regulize = K.regularizers.l2(lambtha)
    output = K.layers.Dense(layers[0], activation=activations[0],
                            input_shape=(nx,),
                            kernel_regularizer=regulize,
                            name='dense')(inputs)

    for i in range(1, len(layers)):
        drop = (K.layers.Dropout(1 - keep_prob))(output)
        output = K.layers.Dense(layers[i], activation=activations[i],
                                kernel_regularizer=regulize,
                                name=('dense_' + str(i)))(drop)

    model = K.Model(inputs=inputs, outputs=output)
    return (model)
