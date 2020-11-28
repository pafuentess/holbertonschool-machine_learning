#!/usr/bin/env python3
""" doc """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """ doc """

    x = keras.Input(shape=(input_dims,))

    # encoder
    layer = keras.layers.Dense(units=hidden_layers[0],
                               activation='relu',
                               input_shape=(input_dims,))(x)

    for i in range(1, len(hidden_layers)):
        layer = keras.layers.Dense(units=hidden_layers[i],
                                   activation='relu')(layer)

    regu = keras.regularizers.l1(lambtha)
    layer = layer = keras.layers.Dense(units=latent_dims,
                                       activity_regularizer=regu,
                                       activation='relu')(layer)

    encoder = keras.Model(inputs=x, outputs=layer)

    # decoder
    dec_x = keras.Input(shape=(latent_dims,))

    layer_dec = keras.layers.Dense(units=hidden_layers[-1],
                                   activation='relu',
                                   input_shape=(latent_dims,))(dec_x)

    for i in range(len(hidden_layers) - 2, -1, -1):
        layer_dec = keras.layers.Dense(units=hidden_layers[i],
                                       activation='relu')(layer_dec)

    layer_dec = keras.layers.Dense(units=input_dims,
                                   activation='sigmoid')(layer_dec)

    decoder = keras.Model(inputs=dec_x, outputs=layer_dec)

    # autoencoder
    bottleneck = encoder(x)
    output = decoder(bottleneck)

    Autoencoder = keras.Model(inputs=x, outputs=output)

    Autoencoder.compile(optimizer=keras.optimizers.Adam(),
                        loss='binary_crossentropy')

    return encoder, decoder, Autoencoder
