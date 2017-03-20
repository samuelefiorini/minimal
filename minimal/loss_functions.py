"""The implemented loss functions and their gradients."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import numpy as np

__losses__ = ('square', 'logit', 'logistic')
__all__ = ('__losses__', 'square_loss', 'square_loss_grad', 'logit_loss',
           'logit_loss_grad')


def square_loss(X, Y, W):
    """Compute the value of the square loss on W.

    When Y is a row/column vector the return ||XW - Y||^2_2, for multi-output
    return ||XW - Y||^2_F.

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
    Yshape = Y.shape
    if len(Yshape) == 1 or (len(Yshape) == 2 and Y.shape[1] == 1):
        order = 2
    else:
        order = 'fro'
    norm = np.linalg.norm(np.dot(X, W) - Y, ord=order)
    return (1.0 / X.shape[0]) * (norm * norm)


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

    Use this function only for binary classification.
    This is the negative loglikelihood.

    Parameters
    ----------
    X : (n, d) float ndarray
        data matrix
    Y : (n,) or (n,1) float ndarray
        labels matrix
    W : (d,) or (d,1) float ndarray
        weights

    Returns
    ----------
    obj : float
        loss function on W
    """
    Yshape = Y.shape
    if len(Yshape) == 1 or (len(Yshape) == 2 and Y.shape[1] > 1):
        raise NotImplementedError('multi-category classification not '
                                  'implemented yet')
    else:
        Xw = X.dot(W)
        return -(1.0 / X.shape[0]) * np.sum(Y * Xw - np.log10(1 + np.exp(-Xw)))
        # return -(1.0 / X.shape[0]) * np.sum(np.log10(1 + np.exp(-Y*X.dot(W))))


def logit_loss_grad(X, Y, W):
    """Compute the logit loss gradient at W.

    Parameters
    ----------
    X : (n, d) float ndarray
        data matrix
    Y : (n,) or (n,1) float ndarray
        labels matrix
    W : (d,) or (d,1) float ndarray
        weights

    Returns
    ----------
    G : (d, T) float ndarray
        logit loss gradient evaluated on W
    """
    Yshape = Y.shape
    if len(Yshape) == 1 or (len(Yshape) == 2 and Y.shape[1] > 1):
        raise NotImplementedError('multi-category classification not '
                                  'implemented yet')
    else:
        p = np.exp(-Y * X.dot(W))
        return -(1.0 / X.shape[0]) * X.T.dot(Y*p/(1+p))
