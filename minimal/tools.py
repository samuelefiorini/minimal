"""Utils module for minimal.

This module contains the utility functions for the actual algorithms.
"""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

from __future__ import division

import numpy as np

from minimal.loss_functions import __losses__
from minimal.loss_functions import square_loss, logit_loss
from minimal.penalties import __penalties__
from minimal.penalties import trace, l21, group_lasso
from functools import partial

__all__ = ['squared', 'get_lipschitz', 'objective_function']


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
        raise NotImplementedError('loss must be in {} '.format(__losses__))


def objective_function(data, labels, W, loss, tau, penalty, groups=None):
    """Evaluate the objective function at a given point.

    This function evaluates the objective function loss(Y, XW) + tau ||W||_*.

    Parameters
    ----------
    data : (n, d) float ndarray
        data matrix
    labels : (n, T) float ndarray
        labels matrix
    loss : string in {'square', 'logit'}
        the selected loss function, this could be either 'square' or 'logit'
        ('logit' not yet implemeted)
    penalty : string in {'trace', 'l21', 'l21_lfro'}
        the penalty to be used, this could be 'trace' for
        nuclear-norm-penalized problems, 'l21' for mixed-norm and 'group-lasso'
        or 'gl' for group lasso
    groups : list of lists (used only for group-lasso)
        the outer list represents the groups and the
        inner lists represent the variables in the groups. E.g. [[1, 2],
        [2, 3]] contains two groups ([1, 2] and [2, 3]) with variable 1 and
        2 in the first group and variables 2 and 3 in the second group.

    Returns
    ----------
    obj : float
        the value of the objective function at a given point
    """
    # Check loss
    if loss.lower() == 'square':
        lossfun = square_loss
    elif loss.lower() in ('logit', 'logistic'):
        lossfun = logit_loss

    # Check penalty
    if penalty.lower() == 'trace':
        penaltyfun = trace
    elif penalty.lower() == 'l21':
        # L21 is the sum of the Euclidean norms of of the rows of the matrix
        penaltyfun = l21
    elif penalty.lower() in ('group-lasso', 'gl'):
        penaltyfun = partial(group_lasso, groups=groups)
    else:
        raise NotImplementedError('loss must be in {} and  penalty '
                                  'in {}'.format(__losses__, __penalties__))

    return lossfun(data, labels, W) + tau * penaltyfun(W)


def regularization_path(minimization_algorithm, data, labels, tau_range,
                        loss='square', penalty='trace', **kwargs):
    """Solution of a penalized VVR problem with warm restart.

    Parameters
    ----------
    minimization_algorithm : callable
        the algorithm of choice
        e.g.: trace_norm_minimization, accelerated_trace_norm_minimization
        l21_norm_minimization or accelerated_l21_norm_minimization
    data : (n, d) float ndarray
        training data matrix
    labels : (n, T) float ndarray
        training labels matrix
    tau_range : (n_tau, ) float ndarray
        range of regularization parameters
    loss : string in {'square', 'logit'}
        the selected loss function, this could be either 'square' or 'logit'
        ('logit' not yet implemeted)
    penalty : string in {'trace', 'l21', 'l21_lfro'}
        the penalty to be used, this could be 'trace' for
        nuclear-norm-penalized problems, 'l21' for multi-task lasso and
        'l21_lfro' for multi-task elastic-net.
    **kwargs : dictionary of keyword-only arguments
        the input list of arguments fed to the minization algorithm of choice

    Returns
    -------
    W_list : (n_tau, ) list of (d, T) matrices
        the solutions corresponding to the input tau_range
    obj_list : (n_tau, ) list of float
        the values of the objective function at convergence
    iter_list : (n_tau, ) list of float
        the number of iterations corresponding to each tau
    """
    # Define output containers
    W_list = list()
    obj_list = list()
    iter_list = list()

    # Initialize Wstart at 0 (then, warm restart)
    Wstart = None

    # Evaluate the tau grid
    for tau in tau_range:
        W, obj, k = minimization_algorithm(data, labels, tau, Wstart,
                                           loss, penalty, return_iter=True,
                                           **kwargs)
        W_list.append(W)
        obj_list.append(obj)
        iter_list.append(k)
        Wstart = W.copy()

    return W_list, obj_list, iter_list
