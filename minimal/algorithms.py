#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Internal algorithms implementations.

This module contains the functions strictly related with the statistical
elaboration of the data.
"""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import sys
import numpy as np


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
    return 2 * np.dot(X.T, (np.dot(X, W) - Y))


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
        return np.linalg.norm(data, ord=2)  # the largest singular value
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
    U, S, V = np.linalg.svd(W)
    soft_thresh_S = np.diag(soft_thresholding(np.diag(S), alpha))
    return np.dot(U, np.dot(soft_thresh_S, V.T))


def trace_norm_minimization(data, labels, tau,
                            loss='square', max_iter=1000):
    """Solving trace-norm penalized vector-valued regression problems.

    Comput the solution of the learning problem

                min loss(Y, XW) + tau ||W||_*
                 W

    where the samples are stored row-wise in X, W is the matrix to be learned
    and each column of Y correspond to a single task. The objective function is
    minimized by means of proximal gradient method (aka forward-backward
    splitting) [REF NEEDED].

    [REF HERE]

    Parameters
    ----------
    data : (n, d) float ndarray
        data matrix
    labels : (n, T) float ndarray
        labels matrix
    tau : float
        regularization parameter
    loss : string
        the selected loss function in {'square', 'logit'}
    max_iter : int
        maximum number of iterations

    Returns
    -------
    W : (d, T) ndarray
        vector-valued regression weights
    """
    if loss == 'square':
        grad = square_loss_grad
    else:
        print('Only square loss implemented.')
        sys.exit(-1)

    # Estimate the fixed step size
    gamma = 1.0 / get_lipschitz(data, loss)

    # Define starting point
    Wk = np.zeros(d, T)

    # Start iterative method
    for k in range(max_iter):

        # Compute proximal gradient step
        W_next = trace_norm_prox(Wk - gamma * grad(data, labels, Wk),
                                 alpha=tau*gamma)

        if ### stopping criterion

        Wk = np.array(W_next)


    return Wk
