#!/usr/bin/env python3
"""Module that creates placeholders for a neural network"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Creates two placeholders, x and y, for the neural network
    
    Args:
        nx (int): number of feature columns in our data
        classes (int): number of classes in our classifier
        
    Returns:
        tuple: (x, y) - placeholders for the neural network
            x is the placeholder for the input data
            y is the placeholder for the one-hot labels
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    
    return x, y