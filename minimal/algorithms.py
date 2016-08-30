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


def trace_norm_minimization(data, labels, tau,
                            loss='square', tol=1e-5, max_iter=50000,
                            return_iter=False):
    """Solution of trace-norm penalized vector-valued regression problems.

    Comput the solution of the learning problem

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
    k : int
        the number of iterations (if return_iter True)
    """
    if loss.lower() == 'square':
        grad = square_loss_grad
    else:
        print('Only square loss implemented so far.')
        sys.exit(-1)

    # Get problem size
    n, d = data.shape
    _, T = labels.shape

    # Estimate the fixed step size
    gamma = 1.0 / get_lipschitz(data, loss)
    print("Step size: {}".format(gamma))

    # Define starting point
    Wk = np.zeros((d, T))
    objk = np.finfo(np.float64).max  # the largest possible value

    obj_list = list()

    # Start iterative method
    for k in range(max_iter):
        # Compute proximal gradient step
        W_next = trace_norm_prox(Wk - gamma * grad(data, labels, Wk),
                                 alpha=tau*gamma)

        # Compute the value of the objective function
        obj_next = objective_function(data, labels, W_next, loss)

        # Check stopping criterion
        if np.abs((objk - obj_next) / objk) <= tol:
            break
        else:
            objk = obj_next
            obj_list.append(objk)
            Wk = np.array(W_next)

    if return_iter:
        return Wk, obj_list, k
    else:
        return Wk, obj_list


def accelerated_trace_norm_minimization(data, labels, tau,
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
    k : int
        the number of iterations (if return_iter True)
    """
    if loss.lower() == 'square':
        grad = square_loss_grad
    else:
        print('Only square loss implemented so far.')
        sys.exit(-1)

    # Get problem size
    n, d = data.shape
    _, T = labels.shape

    # Estimate the fixed step size
    gamma = 1.0 / get_lipschitz(data, loss)

    # Define starting point
    W_old = np.zeros((d, T))  # handy dummy variable
    Zk = np.zeros((d, T))  # search point
    tk = 1.0
    objk = np.finfo(np.float64).max  # the largest possible value

    obj_list = list()

    # Start iterative method
    for k in range(max_iter):
        # Compute proximal gradient step
        Wk = trace_norm_prox(Zk - gamma * grad(data, labels, Zk),
                             alpha=tau*gamma)

        # Compute the value of the objective function
        obj_next = objective_function(data, labels, Wk, loss)

        # Check stopping criterion
        if np.abs((objk - obj_next) / objk) <= tol:
            break
        else:
            # Save the value of the current objective function
            objk = obj_next
            obj_list.append(objk)

            # FISTA Update
            t_next = (1 + np.sqrt(1 + 4 * tk * tk)) * 0.5
            Zk = Wk + ((tk - 1) / t_next) * (Wk - W_old)

            # Point and search point update
            W_old = np.array(Wk)
            tk = t_next

    if return_iter:
        return Wk, obj_list, k
    else:
        return Wk, obj_list
