#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The implemented penalties and their proximal mapping."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import numpy as np

from minimal.loss_functions import __losses__
from six.moves import map

__penalties__ = ('trace', 'l21', 'group-lasso', 'gl')


def trace_norm_bound(X, Y, loss='square'):
    """Compute maximum value for the trace norm regularization parameter.

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
        # In this case max_tau := 2/n * max_sing_val(X^T * Y)
        return np.linalg.norm(np.dot(X.T, Y), ord=2) * (2.0/X.shape[0])
    else:
        raise NotImplementedError('Loss function must be '
                                  'in {}.'.format(__losses__))


def l21_norm_bound(X, Y, loss='square'):
    """Compute maximum value for the l12-norm regularization parameter.

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
        maximum value for the l21-norm regularization parameter
    """
    if loss.lower() == 'square':
        # In this case max_tau := 2/n * max(||[X^T * Y]s||_2)
        # First compute the 2-norm of each row of X^T * Y
        norm2 = map(lambda x: np.linalg.norm(x, ord=2), X.T.dot(Y))
        return max(norm2) * (2.0/X.shape[0])
    else:
        raise NotImplementedError('Loss function must be '
                                  'in {}.'.format(__losses__))


def group_lasso_norm_bound(X, Y, loss='square'):
    """Compute maximum value for the l12-norm regularization parameter.

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
        maximum value for the l21-norm regularization parameter
    """
    raise NotImplementedError('TODO')


def trace(X):
    """Compute the trace-norm on the input matrix."""
    return np.linalg.norm(X, ord='nuc')


def l21(X):
    """Compute the L21 norm on the input matrix."""
    return sum(map(lambda x: np.linalg.norm(x, ord=2), X))


def group_lasso(x, groups):
    """Compute the group-lasso penalty on the input array with the given groups."""
    return np.linalg.norm([x[g].T.dot(x[g]) for g in groups])


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
    return np.sign(w) * np.clip(np.abs(w) - alpha, 0.0, np.inf)


def euclidean_norm_prox(x, alpha):
    """Compute the proximal mapping for the Euclidean norm."""
    return np.max((1 - alpha/np.linalg.norm(x, ord=2), 0)) * x


def block_soft_thresholding(x, alpha, groups):
    """Compute the proximal mapping for the group-lasso penalty."""
    return np.vstack([euclidean_norm_prox(x[g], alpha) for g in groups])


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


def l21_norm_prox(W, alpha):
    """Compute l2,1-norm proximal operator on W.

    This function returns: prox             (W)
                             alpha ||.||_2,1

    Parameters
    ----------
    W : (n1, n2) float ndarray
        proximal operator input
    alpha : float
        proximal threshold

    Returns
    ----------
    Wt : (n1, n2) float ndarray
        l2,1-norm prox result
    """
    d, T = W.shape

    # Compute the soft-thresholding operator for each row of an unitary matrix
    ones = np.ones(T)
    Wst = np.empty(W.shape)
    for i, Wi in enumerate(W):
        thresh = alpha / np.sqrt(Wi.T.dot(Wi))
        Wst[i, :] = soft_thresholding(ones, thresh)

    # Return the Hadamard-product between Wst and W
    return W * Wst
