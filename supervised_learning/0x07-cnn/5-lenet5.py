#!/usr/bin/env python3
""" doc """

import tensorflow.keras as K


def lenet5(X):
    """ doc """

    initializer = K.initializers.he_normal(seed=None)

    C_layer1 = K.layers.Conv2D(filters=6, kernel_size=5, padding='same',
                               activation='relu',
                               kernel_initializer=initializer)(X)

    P_layer1 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(C_layer1)

    C_layer2 = K.layers.Conv2D(filters=16, kernel_size=5, padding='valid',
                               activation='relu',
                               kernel_initializer=initializer)(P_layer1)

    P_layer2 = K.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(C_layer2)

    flatten = K.layers.Flatten()(P_layer2)

    F_layer1 = K.layers.Dense(units=120, activation='relu',
                              kernel_initializer=initializer)(flatten)

    F_layer2 = K.layers.Dense(units=84, activation='relu',
                              kernel_initializer=initializer)(F_layer1)

    L_output = K.layers.Dense(units=10,
                              kernel_initializer=initializer,
                              activation='softmax')(F_layer2)

    model = K.models.Model(inputs=X, outputs=L_output)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
