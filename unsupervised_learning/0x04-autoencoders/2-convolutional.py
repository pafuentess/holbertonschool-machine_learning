#!/usr/bin/env python3
""" doc """

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ doc """
    inputs = keras.Input(shape=input_dims)

    conv_layer = keras.layers.Conv2D(filters=filters[0],
                                     kernel_size=(3, 3),
                                     padding="same",
                                     activation='relu',
                                     )(inputs)
    max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                            padding='same')(conv_layer)

    for i in range(1, len(filters)):
        conv_layer = keras.layers.Conv2D(filters=filters[i],
                                         kernel_size=(3, 3),
                                         padding="same",
                                         activation='relu',
                                         )(max_pool_2d)
        max_pool_2d = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                padding='same')(conv_layer)

    encoder = keras.Model(inputs=inputs, outputs=max_pool_2d)

    # DECODER
    last_filter = input_dims[-1]
    inputs_dec = keras.Input(shape=latent_dims)
    my_conv_layer_dec = keras.layers.Conv2D(filters=filters[-1],
                                            kernel_size=(3, 3),
                                            padding="same",
                                            activation='relu',
                                            )(inputs_dec)

    upsampling_lay = keras.layers.UpSampling2D(
        size=(2, 2))(my_conv_layer_dec)

    for i in range(len(filters) - 1, 1, -1):
        my_conv_layer_dec = keras.layers.Conv2D(filters=filters[i],
                                                kernel_size=(3, 3),
                                                padding="same",
                                                activation='relu'
                                                )(upsampling_lay)

        upsampling_lay = keras.layers.UpSampling2D(
            size=(2, 2))(my_conv_layer_dec)

    my_conv_layer_dec = keras.layers.Conv2D(filters=filters[0],
                                            kernel_size=(3, 3),
                                            padding="valid",
                                            activation='relu'
                                            )(upsampling_lay)

    upsampling_lay = keras.layers.UpSampling2D(
        size=(2, 2))(my_conv_layer_dec)

    my_conv_layer_dec = keras.layers.Conv2D(filters=last_filter,
                                            kernel_size=(3, 3),
                                            padding="same",
                                            activation='sigmoid'
                                            )(upsampling_lay)

    decoder = keras.Model(inputs=inputs_dec, outputs=my_conv_layer_dec)

    # AUTOENCODER
    auto_bottleneck = encoder(inputs)
    auto_output = decoder(auto_bottleneck)

    auto = keras.Model(inputs=inputs, outputs=auto_output)

    # compilation
    auto.compile(optimizer=keras.optimizers.Adam(),
                 loss='binary_crossentropy')

    return encoder, decoder, auto
