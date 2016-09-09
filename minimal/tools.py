#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utils module for minimal.

This module contains the utility functions for the actual algorithms.
"""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import sys
import numpy as np

__all__ = ['square_loss', 'square_loss_grad', 'get_lipschitz',
           'soft_thresholding', 'trace_norm_bound',
           'objective_function', 'trace_norm_prox']


def squared(X):
    """Return convenient squared matrix.

    Let n, d = X.shape. If n < d this function returns X * X^T else it returns
    X^T * X.

    Parameters
    ----------
    X : (n, d) float ndarray
        input matrix

    Returns
    ----------
    XX : (min(n,d), min(n,d)) float array
        squared input matrix
    """
    n, d = X.shape
    if n <= d:
        return X.dot(X.T)
    else:
        return X.T.dot(X)


def square_loss(X, Y, W):
    """Compute the value of the square loss on W.

    Parameters
    ----------
    X : (n, d) float ndarray
        data matrix
    Y : (n, T) float ndarray
        labels matrix
    W : (d, T) float ndarray
        weights

    Returns
    ----------
    obj : float
        The value of the objective function on W.
    """
    return 1.0 / X.shape[0] * np.linalg.norm(np.dot(X, W) - Y, ord='fro') ** 2


def square_loss_grad(X, Y, W):
    """Compute the square loss gradient at W.

    Parameters
    ----------
    X : (n, d) float ndarray
        data matrix
    Y : (n, T) float ndarray
        labels matrix
    W : (d, T) float ndarray
        weights

    Returns
    ----------
    G : (d, T) float ndarray
        square loss gradient evaluated on the current iterate W.
    """
    return 2.0 / X.shape[0] * np.dot(X.T, (np.dot(X, W) - Y))


def get_lipschitz(data, loss):
    """Get the Lipschitz constant for a specific loss function.

    Parameters
    ----------
    data : (n, d) float ndarray
        data matrix
    loss : string
        the selected loss function in {'square', 'logit'}

    Returns
    ----------
    L : float
        the Lipschitz constant
    """
    if loss == 'square':
        # compute the largest singular value
        return np.linalg.norm(squared(data), ord=2)
    else:
        print('Only square loss implemented.')
        sys.exit(-1)


def soft_thresholding(w, alpha):
    """Compute the element-wise soft-thresholding operator on the vector w.

    Parameters
    ----------
    w : (d,) or (d, 1) ndarray
        input vector
    alpha : float
        threshold

    Returns
    ----------
    wt : (d,) or (d, 1) ndarray
        soft-thresholded vector
    """
    return np.sign(w) * np.clip(np.abs(w) - alpha, 0, np.inf)


def trace_norm_bound(X, Y, loss='square'):
    """Compute maximum value for the trace norm parameter.

    Parameters
    ----------
    data : (n, d) float ndarray
        data matrix
    labels : (n, T) float ndarray
        labels matrix
    loss : string
        the selected loss function in {'square', 'logit'}. Default is 'square'

    Returns
    ----------
    max_tau : float
        maximum value for the trace norm regularization parameter
    """
    if loss.lower() == 'square':
        # In this case max_tau := 2/n * max_sing_val(X^T Y)
        return np.linalg.norm(np.dot(X.T, Y), ord=2) * (2.0/X.shape[0])
    else:
        print('Only square loss implemented so far.')
        sys.exit(-1)


def objective_function(data, labels, W, loss='square'):
    """Evaluate the objective function at a given point.

    This function evaluates the objective function loss(Y, XW) + tau ||W||_*.

    Parameters
    ----------
    data : (n, d) float ndarray
        data matrix
    labels : (n, T) float ndarray
        labels matrix
    loss : string
        the selected loss function in {'square', 'logit'}. Default is 'square'

    Returns
    ----------
    obj : float
        the value of the objective function at a given point
    """
    if loss.lower() == 'square':
        fun = square_loss
        return fun(data, labels, W) + np.linalg.norm(W, ord='nuc')
    else:
        print('Only square loss implemented so far.')
        sys.exit(-1)


def trace_norm_prox(W, alpha):
    """Compute trace norm proximal operator on W.

    This function returns: prox           (W)
                             alpha ||.||_*

    Parameters
    ----------
    W : (n1, n2) float ndarray
        proximal operator input
    alpha : float
        proximal threshold

    Returns
    ----------
    Wt : (n1, n2) float ndarray
        trace norm prox result
    """
    d, T = W.shape
    U, s, V = np.linalg.svd(W, full_matrices=True)
    # U ~ (d, d)
    # s ~ (min(d, T), min(d, T))
    # V ~ (T, T)
    s = soft_thresholding(s, alpha)
    # make the output n1 x n2
    if d >= T:
        st_S = np.vstack((np.diag(s), np.zeros((np.abs(d-T), T))))
    else:
        st_S = np.hstack((np.diag(s), np.zeros((d, np.abs(d-T)))))
    return np.dot(U, np.dot(st_S, V))
