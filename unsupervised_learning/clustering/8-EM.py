#!/usr/bin/env python3

"""
This module contains a function that perfoms
expectation maximization for a GMM
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k,
                             iterations=1000, tol=1e-5, verbose=False):
    """
    initializes variables for a Gaussian Mixture Model

    X: numpy.ndarray (n, d) containing the dataset
        - n no. of data points
        - d no. of dimensions for each data point
    k: positive integer containing the number of clusters
    iterations: positive integer containing the maximum number of iterations
    tol: non-negative float containing tolerance of the log likelihood
    verbose: boolean that determines if output should be printed
    returns:
        pi, m, S, g, log_like or None, None, None, None, None on failure
        - pi: numpy.ndarray (k,) containing the priors for each cluster
        - m: numpy.ndarray (k, d) containing centroid means for each cluster
        - S: numpy.ndarray (k, d, d) covariance matrices for each cluster
        - g: numpy.ndarray (k, n) containing the posterior
            probabilities for each data point in each cluster
        - log_like: log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, log_like = expectation(X, pi, m, S)
    prev_like = log_like
    msg = "Log Likelihood after {} iterations: {}"

    if verbose:
        print(msg.format(0, log_like.round(5)))

    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        g, total_log_like = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            print(msg.format(i, total_log_like.round(5)))
        if abs(prev_like - total_log_like) <= tol:
            break
        prev_like = total_log_like
        log_like = total_log_like

    if verbose and (i % 10 != 0):
        print(msg.format(i, total_log_like.round(5)))

    return pi, m, S, g, total_log_like
