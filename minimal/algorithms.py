#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Internal algorithms implementations for trace-norm penalized problems.

This module contains the functions strictly related with the statistical
elaboration of the data.
"""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

from __future__ import division
import sys
import numpy as np
from sklearn.utils import deprecated
from collections import deque
from . import tools


@deprecated('Use minimal.optimization.ISTA() instead.')
def trace_norm_minimization(data, labels, tau, Wstart=None,
                            loss='square', tol=1e-5, max_iter=50000,
                            return_iter=False):
    """Solution of trace-norm penalized vector-valued regression problems.

    Compute the solution of the learning problem

                min loss(Y, XW) + tau ||W||_*
                 W

    where the samples are stored row-wise in X, W is the matrix to be learned
    and each column of Y correspond to a single task. The objective function is
    minimized by means of proximal gradient method (aka forward-backward
    splitting) [!!REF NEEDED!!].

    [!!REF HERE!!]

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
        the selected loss function in {'square', 'logit'}. Default is 'square'
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
    if loss.lower() == 'square':
        grad = tools.square_loss_grad
    else:
        print('Only square loss implemented so far.')
        sys.exit(-1)

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
        W_next = tools.trace_norm_prox(Wk - gamma * grad(data, labels, Wk),
                                       alpha=tau*gamma)

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


@deprecated('Use minimal.optimization.FISTA() instead.')
def accelerated_trace_norm_minimization(data, labels, tau, Wstart=None,
                                        loss='square', tol=1e-5,
                                        max_iter=50000,
                                        return_iter=False):
    """Fast solution of trace-norm penalized vector-valued regression problems.

    Compute the solution of the learning problem

                min loss(Y, XW) + tau ||W||_*
                 W

    where the samples are stored row-wise in X, W is the matrix to be learned
    and each column of Y correspond to a single task. The objective function is
    minimized by means of accelerated proximal gradient method (aka
    forward-backward splitting). This implementation extends to the matrix case
    the FISTA algorithm presented in [1]. Another relevant reference for this
    work is [2], but they use a FISTA-like acceleration along with an adaptive
    step-size defined via backtracking (aka line search). Such trick is not
    implemented here.

    References
    ----------
    [1] Beck, Amir, and Marc Teboulle. "A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems." SIAM journal on imaging sciences
    2.1 (2009): 183-202.
    [2] Ji, Shuiwang, and Jieping Ye. "An accelerated gradient method for trace
    norm minimization." Proceedings of the 26th annual international conference
    on machine learning. ACM, 2009.

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
        the selected loss function in {'square', 'logit'}. Default is 'square'
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
    if loss.lower() == 'square':
        grad = tools.square_loss_grad
    else:
        print('Only square loss implemented so far.')
        sys.exit(-1)

    # Get problem size
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

    # obj_list = list()

    # FIFO list of objective function values. Added to improve FISTA stability
    max_objq_length = 5
    obj_deque = deque([objk], max_objq_length)

    # Start iterative method
    for k in range(max_iter):
        # Compute proximal gradient step
        Wk = tools.trace_norm_prox(Zk - gamma * grad(data, labels, Zk),
                                   alpha=tau*gamma)

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
            # obj_list.append(objk)
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
