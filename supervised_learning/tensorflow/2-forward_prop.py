#!/usr/bin/env python3
"""
Module for forward propagation in tensorflow
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network"""
    prediction = x
    for i in range(len(layer_sizes)):
        prediction = create_layer(prediction, layer_sizes[i], activations[i])
    return prediction