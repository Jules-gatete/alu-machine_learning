#!/usr/bin/env python3
"""
Module to create training operation for neural network
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """Creates training operation using gradient descent optimizer"""
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
