#!/usr/bin/env python3
"""
Module that creates placeholders for neural network
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """Creates placeholders for neural network"""

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    return x, y
