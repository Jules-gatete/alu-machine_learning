#!/usr/bin/env python3
"""
Module that creates a layer for neural network
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates a layer for neural network"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        name='layer'
    )
    return layer(prev)
