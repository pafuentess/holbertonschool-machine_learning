#!/usr/bin/env python3

from matplotlib import pyplot
import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """ doc """
    X_p = K.applications.resnet50.preprocess_input(X)
    X_y = K.utils.to_categorical(Y, 10)
    return(X_p, X_y)


def resize(X):
    """ resize """
    return K.backend.resize_images(X, 7, 7,
                                   data_format="channels_last",
                                   interpolation='bilinear')


if __name__ == "__main__":
    (Xtrain, Ytrain), (Xtest, Ytest) = K.datasets.cifar10.load_data()
    Xtrain, Ytrain = preprocess_data(Xtrain, Ytrain)
    Xtest, Ytest = preprocess_data(Xtest, Ytest)

    ResNet50_model = K.applications.ResNet50(weights='imagenet',
                                             include_top=False,
                                             input_shape=(224, 224, 3))

    ResNet50_model.trainable = False
    Input = K.Input(shape=(32, 32, 3))
    resizeImage = K.layers.Lambda(resize)(Input)
    x = ResNet50_model(resizeImage, training=False)
    x = K.layers.Flatten()(x)
    x = K.layers.Dense(1000, activation='relu')(x)
    x = K.layers.Dropout(0.2)(x)
    x = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(Input, x)
    opt = K.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpoint = K.callbacks.ModelCheckpoint(save_best_only=True, mode="max",
                                             monitor="val_acc",
                                             filepath="cifar10.h5")

    model.fit(Xtrain, Ytrain, epochs=5, batch_size=224,
              validation_data=(Xtest, Ytest),
              verbose=1, callbacks=[checkpoint])
    model.save("cifar10.h5")
