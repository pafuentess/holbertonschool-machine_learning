#!/usr/bin/env python3
""" doc """

import tensorflow as tf
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def calculate_loss(y, y_pred):
    """ doc """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def calculate_accuracy(y, y_pred):
    """ doc """
    y_max = tf.argmax(y, axis=1)
    y_pred_max = tf.argmax(y_pred, axis=1)
    equal = tf.equal(y_max, y_pred_max)
    cast = tf.cast(equal, tf.float32)
    accuar = tf.reduce_mean(cast)
    return (accuar)


def create_layer(prev, n, activation):
    """ doc """
    initialize = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=initialize,
                            activation=activation, name="layer")
    return (layer(prev))


def create_placeholders(nx, classes):
    """ doc """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return (x, y)


def forward_prop(x, layer_sizes=[], activations=[]):
    """ doc """
    layer = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        if i != len(layer_sizes) - 1:
            layer = create_batch_norm_layer(layer,
                                            layer_sizes[i], activations[i])
        else:
            layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ doc """
    optimize = tf.train.AdamOptimizer(learning_rate=alpha,
                                      beta1=beta1, beta2=beta2,
                                      epsilon=epsilon)
    train = optimize.minimize(loss)
    return train


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ doc """
    train = tf.train.inverse_time_decay(alpha, decay_rate=decay_rate,
                                        global_step=global_step,
                                        decay_steps=decay_step,
                                        staircase=True)
    return train


def create_batch_norm_layer(prev, n, activation):
    """ doc """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=None, kernel_initializer=init)

    z = layer(prev)

    m, s = tf.nn.moments(z, axes=[0])
    beta = tf.Variable(tf.zeros([z.get_shape()[-1]]))
    gamma = tf.Variable(tf.ones([z.get_shape()[-1]]))
    Znorm = tf.nn.batch_normalization(z, m, s, beta, gamma, 1e-8)

    return (activation(Znorm))


def shuffle_data(X, Y):
    """ doc """
    permutation = np.random.permutation(len(Y))
    Xp = X[permutation]
    Yp = Y[permutation]
    return (Xp, Yp)


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """ doc """
    x, y = create_placeholders(Data_train[0].shape[1], Data_train[1].shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        X_train = Data_train[0]
        Y_train = Data_train[1]
        X_valid = Data_valid[0]
        Y_valid = Data_valid[1]

        iteration = X_train.shape[0] / batch_size
        iterations = int(iteration)
        if iteration > iterations:
            iterations = int(iteration) + 1
            extra = True
        else:
            extra = False

        for i in range(epochs + 1):
            cost_train, accuracy_train = sess.run(
                    [loss, accuracy],
                    feed_dict={x: X_train, y: Y_train})
            cost_valid, accuracy_valid = sess.run(
                    [loss, accuracy],
                    feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_valid))
            print("\tValidation Accuracy: {}".format(accuracy_valid))

            if i < epochs:
                Xsh, Ysh = shuffle_data(X_train, Y_train)
                batch = Xsh.shape[0]
                start = 0
                step = 1
                while batch > 0:
                    if batch - batch_size < 0:
                        end = Xsh.shape[0]
                    else:
                        end = start + batch_size
                    X = Xsh[start:end]
                    Y = Ysh[start:end]

                    sess.run(train_op, feed_dict={x: X, y: Y})

                    if step % 100 == 0:
                        cost_train = sess.run(loss, feed_dict={x: X, y: Y})
                        accu_train = sess.run(accuracy, feed_dict={x: X, y: Y})
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(cost_train))
                        print("\t\tAccuracy: {}".format(accu_train))
                    step = step + 1
                    batch = batch - batch_size
                    start = start + batch_size
            sess.run(tf.assign(global_step, global_step + 1))

        save_path = saver.save(sess, save_path)
        return save_path
