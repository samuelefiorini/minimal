"""The implemented loss functions and their gradients."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import numpy as np

__losses__ = ('square', 'logit', 'logistic')


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


def logit_loss(X, Y, W):
    """Compute the value of the logit loss on W.

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
        loss function on W
    """
    raise NotImplementedError('TODO')


def logit_loss_grad(X, Y, W):
    """Compute the logit loss gradient at W.

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
        logit loss gradient evaluated on W
    """
    raise NotImplementedError('TODO')
