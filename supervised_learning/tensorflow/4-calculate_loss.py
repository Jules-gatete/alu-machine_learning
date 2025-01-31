#!/usr/bin/env python3
"""
Module to calculate softmax cross-entropy loss
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction"""
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=y,
            logits=y_pred))
