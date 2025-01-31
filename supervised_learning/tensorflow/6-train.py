#!/usr/bin/env python3
"""
Module to build, train and save neural network classifier
"""
import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """Builds, trains, and saves a neural network classifier"""
    nx = X_train.shape[1]
    classes = Y_train.shape[1]

    x, y = create_placeholders(nx, classes)
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    # Add to collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Initialize variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        # Print initial state
        t_cost, t_accuracy = sess.run([loss, accuracy],
                                    feed_dict={x: X_train, y: Y_train})
        v_cost, v_accuracy = sess.run([loss, accuracy],
                                    feed_dict={x: X_valid, y: Y_valid})
        print("After 0 iterations:")
        print(f"\tTraining Cost: {t_cost}")
        print(f"\tTraining Accuracy: {t_accuracy}")
        print(f"\tValidation Cost: {v_cost}")
        print(f"\tValidation Accuracy: {v_accuracy}")

        # Training loop
        for i in range(1, iterations + 1):
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

            if i % 100 == 0 or i == iterations:
                t_cost, t_accuracy = sess.run([loss, accuracy],
                                            feed_dict={x: X_train, y: Y_train})
                v_cost, v_accuracy = sess.run([loss, accuracy],
                                            feed_dict={x: X_valid, y: Y_valid})
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {t_cost}")
                print(f"\tTraining Accuracy: {t_accuracy}")
                print(f"\tValidation Cost: {v_cost}")
                print(f"\tValidation Accuracy: {v_accuracy}")

        return saver.save(sess, save_path)
