#!/usr/bin/env python3
"""This module contains a function that perfoms
finds the best number of clusters for a GMM using the
Bayesian Information Criterion"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    finds the best number of clusters for a GMM using the
    Bayesian Information Criterion
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None, None, None

    if not isinstance(iterations, int):
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = 10
    likelihoods = []
    bics = []
    results = []
    for k in range(kmin, kmax + 1):
        try:
            pi, m, S, g, ll = expectation_maximization(X, k, iterations, tol, verbose)
        except Exception:
            likelihoods.append(None)
            bics.append(None)
            results.append(None)
            continue
        # Number of parameters: weights (k-1), means (k*d), covariances (k*d*(d+1)/2)
        p = (k - 1) + k * d + k * d * (d + 1) / 2
        bic = p * np.log(n) - 2 * ll
        likelihoods.append(ll)
        bics.append(bic)
        results.append((pi, m, S))
    # Find best k (lowest BIC, ignoring failed runs)
    valid = [(i, b) for i, b in enumerate(bics) if b is not None]
    if not valid:
        return None, None, None, None
    best_idx = min(valid, key=lambda x: x[1])[0]
    best_k = kmin + best_idx
    best_res = results[best_idx]
    return best_k, best_res, np.array(likelihoods, dtype=object), np.array(bics, dtype=object)
