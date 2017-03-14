#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Internal algorithms implementations for penalized minimization problems.

This module contains implementations of proximal gradient methods such as
ISTA [1] and FISTA [1].

References
----------
[1] Beck, Amir, and Marc Teboulle. "A fast iterative shrinkage-thresholding
algorithm for linear inverse problems." SIAM journal on imaging sciences
2.1 (2009): 183-202.
[2] Ji, Shuiwang, and Jieping Ye. "An accelerated gradient method for trace
norm minimization." Proceedings of the 26th annual international conference
on machine learning. ACM, 2009.
"""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import numpy as np

from collections import deque
from functools import partial
from minimal import tools
from minimal.loss_functions import __losses__
from minimal.loss_functions import square_loss_grad, logit_loss_grad
from minimal.penalties import (trace_norm_prox, l21_norm_prox,
                               block_soft_thresholding)
from minimal.penalties import __penalties__

__all__ = ('ISTA', 'FISTA', 'get_prox')


def get_prox(penalty='trace', groups=None):
    """Get the proximal mapping function corresponding to the penalty.

    Parameters
    ----------
    penalty : string
        the selected penalty function in {'l21', 'trace', 'group-lasso'};
        default is 'trace'.
    groups : array-like (used only for group-lasso)
        [[1,2,3], [4,5,6], ...]

    Returns
    -------
    prox : function
        the corresponding proximal mapping
    """
    # Load the penalty
    if penalty.lower() == 'trace':
        prox = trace_norm_prox
    elif penalty.lower() == 'l21':
        prox = l21_norm_prox
    elif penalty.lower() in ('group-lasso', 'gl'):
        prox = partial(block_soft_thresholding, groups=groups)
    else:
        raise NotImplementedError('penalty must be '
                                  'in {}.'.format(__penalties__))
    return prox


def ISTA(data, labels, tau, Wstart=None, loss='square', penalty='trace',
         tol=1e-5, max_iter=10000, return_iter=False):
    """Iterative-Shrinking Thresholding Algorithm.

    Compute the solution of the minimization problem

                min loss(Y, XW) + tau pen(W)
                 W

    via forward-backward splitting.

    The samples are stored row-wise in X, W is the weight matrix to
    learn and each column of Y correspond to a single task (for vector-valued
    regression) or a binary entry (for multi-category classification).

    Parameters
    ----------
    data : (n, d) float ndarray
        data matrix
    labels : (n, T) float ndarray
        labels matrix
    tau : float
        regularization parameter
    Wstart : (d, T) float array
        starting point for the iterative minization algorithm
    loss : string
        the selected loss function in {'square', 'logit'}; default is 'square'.
    penalty : string
        the selected penalty function in {'l21', 'trace', 'group-lasso'};
        default is 'trace'.
    tol : float
        stopping rule tolerance. Default is 1e-5.
    max_iter : int
        maximum number of iterations. Default is 1e4.
    return_iter : bool
        return the number of iterations before convergence

    Returns
    -------
    W : (d, T) ndarray
        vector-valued regression weights
    obj : float
        the value of the objective function at convergence
    k : int
        the number of iterations (if return_iter True)
    """
    # Load the loss function
    if loss.lower() == 'square':
        grad = square_loss_grad
    elif loss.lower() in ('logit', 'logistic'):
        grad = logit_loss_grad
    else:
        raise NotImplementedError('loss must be '
                                  'in {}.'.format(__losses__))

    # Load the penalty
    if penalty.lower() == 'trace':
        prox = trace_norm_prox
    elif penalty.lower() == 'l21':
        prox = l21_norm_prox
    elif penalty.lower() in ('group-lasso', 'gl'):
        prox = block_soft_thresholding
    else:
        raise NotImplementedError('penalty must be '
                                  'in {}.'.format(__penalties__))

    # Get problem size
    n, d = data.shape
    _, T = labels.shape

    # Estimate the fixed step size
    gamma = 1.0 / tools.get_lipschitz(data, loss)

    # Define starting point
    if Wstart is None:
        Wk = np.zeros((d, T))
    else:
        Wk = Wstart
    objk = np.finfo(np.float64).max  # the largest possible value

    obj_list = list()

    # Start iterative method
    for k in range(max_iter):
        # Compute proximal gradient step
        W_next = prox(Wk - gamma * grad(data, labels, Wk), alpha=tau*gamma)

        # Compute the value of the objective function
        obj_next = tools.objective_function(data, labels, W_next, loss)

        # Check stopping criterion
        if np.abs((objk - obj_next) / objk) <= tol:
            break
        else:
            objk = obj_next
            obj_list.append(objk)
            Wk = np.array(W_next)

    if return_iter:
        return Wk, obj_list[-1], k
    else:
        return Wk, obj_list[-1]


def FISTA(data, labels, tau, Wstart=None, loss='square', penalty='trace',
          tol=1e-5, max_iter=10000, return_iter=False):
    """Fast Iterative-Shrinking Thresholding Algorithm.

    Compute the solution of the minimization problem

                min loss(Y, XW) + tau pen(W)
                 W

    via acceperated forward-backward splitting.

    The samples are stored row-wise in X, W is the weight matrix to
    learn and each column of Y correspond to a single task (for vector-valued
    regression) or a binary entry (for multi-category classification).

    Parameters
    ----------
    data : (n, d) float ndarray
        data matrix
    labels : (n, T) float ndarray
        labels matrix
    tau : float
        regularization parameter
    Wstart : (d, T) float array
        starting point for the iterative minization algorithm
    loss : string
        the selected loss function in {'square', 'logit'}; default is 'square'.
    penalty : string
        the selected penalty function in {'l21', 'trace', 'group-lasso'};
        default is 'trace'.
    tol : float
        stopping rule tolerance. Default is 1e-5.
    max_iter : int
        maximum number of iterations. Default is 1e4.
    return_iter : bool
        return the number of iterations before convergence

    Returns
    -------
    W : (d, T) ndarray
        vector-valued regression weights
    obj : float
        the value of the objective function at convergence
    k : int
        the number of iterations (if return_iter True)
    """
    # Load the loss function
    if loss.lower() == 'square':
        grad = square_loss_grad
    elif loss.lower() in ('logit', 'logistic'):
        grad = logit_loss_grad
    else:
        raise NotImplementedError('loss must be '
                                  'in {}.'.format(__losses__))

    # Load the penalty
    if penalty.lower() == 'trace':
        prox = trace_norm_prox
    elif penalty.lower() == 'l21':
        prox = l21_norm_prox
    elif penalty.lower() in ('group-lasso', 'gl'):
        prox = block_soft_thresholding
    else:
        raise NotImplementedError('penalty must be '
                                  'in {}.'.format(__penalties__))

    n, d = data.shape
    _, T = labels.shape

    # Estimate the fixed step size
    gamma = 1.0 / tools.get_lipschitz(data, loss)

    # Define starting point
    W_old = np.zeros((d, T))  # handy dummy variable
    # Define starting point
    if Wstart is None:
        Zk = np.zeros((d, T))
    else:
        Zk = Wstart
    tk = 1.0
    objk = np.finfo(np.float64).max  # the largest possible value

    # FIFO list of objective function values. Added to improve FISTA stability
    max_objq_length = 5
    obj_deque = deque([objk], max_objq_length)

    # Start iterative method
    for k in range(max_iter):
        # Compute proximal gradient step
        Wk = prox(Zk - gamma * grad(data, labels, Zk), alpha=tau*gamma)

        # Compute the value of the objective function
        obj_next = tools.objective_function(data, labels, Wk, loss)

        # Check stopping criterion
        # if np.abs((objk - obj_next) / objk) <= tol:
        obj_mean = sum(obj_deque) / len(obj_deque)
        if np.abs((obj_mean - obj_next) / obj_mean) <= tol:
            break
        else:
            # Save the value of the current objective function
            objk = obj_next
            obj_deque.append(obj_next)

            # FISTA Update
            t_next = (1 + np.sqrt(1 + 4 * tk * tk)) * 0.5
            Zk = Wk + ((tk - 1) / t_next) * (Wk - W_old)

            # Point and search point update
            W_old = np.array(Wk)
            tk = t_next

    if return_iter:
        return Wk, obj_deque[-1], k
    else:
        return Wk, obj_deque[-1]
