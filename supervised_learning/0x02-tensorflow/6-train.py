#!/usr/bin/env python3
""" doc """

import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """ doc """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuar = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuar', accuar)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    trainOp = create_train_op(loss, alpha)
    tf.add_to_collection('trainOp', trainOp)

    saver = tf.train.Saver()
    initialize = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(initialize)
        for i in range(iterations + 1):
            cT, aT = sess.run([loss, accuar],
                              feed_dict={x: X_train, y: Y_train})
            cV, aV = sess.run([loss, accuar],
                              feed_dict={x: X_valid, y: Y_valid})
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cT))
                print("\tTraining Accuracy: {}".format(aT))
                print("\tValidation Cost: {}".format(cV))
                print("\tValidation Accuracy: {}".format(aV))
            if i < iterations:
                sess.run(trainOp, feed_dict={x: X_train, y: Y_train})
        save_path = saver.save(sess, save_path)
    return saver_path
