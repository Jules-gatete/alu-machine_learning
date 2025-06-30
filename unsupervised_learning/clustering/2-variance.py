#!/usr/bin/env python3

"""
This module contains a function that
calculates total intra-cluster variance for a dataset
"""

import numpy as np


def variance(X, C):
    """
    calculates intra-cluster variance for a dataset

    X: numpy.ndarray (n, d) containing the dataset that
    will be used for K-means clustering
        - n no. of data points
        - d no. of dimensions for each data point
    C: numpy.ndarray (k, d) containing the centroid
        for each cluster

    return:
        - var: total intra-cluster variance
    """
    if not (isinstance(X, np.ndarray) and isinstance(C, np.ndarray)):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2 or X.shape[1] != C.shape[1]:
        return None
    # Compute squared distances from each point to each centroid
    dists = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    # For each point, find the closest centroid
    min_dists = np.min(dists, axis=0)
    # Total variance is the sum of squared distances to closest centroid
    var = np.sum(min_dists ** 2)
    return var
