#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal core functionalities.

This module contains the functions to perform core functionalities, such as
parameter (model) selection and assessment.
"""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

from __future__ import division
import sys
import multiprocessing as mp
import numpy as np
from minimal.algorithms import regularization_path
from minimal.algorithms import trace_norm_minimization
from minimal.algorithms import accelerated_trace_norm_minimization
from minimal.algorithms21 import l21_norm_minimization
from minimal.algorithms21 import accelerated_l21_norm_minimization
from minimal import tools

# Legacy import
try:
    from sklearn.model_selection import KFold
except:
    from sklearn.cross_validation import KFold


def get_minimizer(algorithm='FISTA', penalty='trace'):
    """Return the minimizer of choice.

    Parameters
    ----------
    algorithm : string in {'ISTA', 'FISTA'}
        the minimization algorithm of choice, this could be either
        'ISTA' (default) or 'FISTA'
    penalty : string in {'trace', 'l21', 'l21_lfro'}
        the penalty to be used, this could be 'trace' for
        nuclear-norm-penalized problems, 'l21' for multi-task lasso and
        'l21_lfro' for multi-task elastic-net.

    Returns
    -------
    minimizer : callable
        the selected minimizer
    bound : callable
        the selected function to evaluate the max tau
    """
    if penalty.lower() == 'trace':
        bound = tools.trace_norm_bound

        # Check minimization algorithm
        if algorithm.lower() == 'ista':
            minimizer = trace_norm_minimization
        elif algorithm.lower() == 'fista':
            minimizer = accelerated_trace_norm_minimization
        else:
            print("Minimization strategy {} not understood, "
                  "it must be 'ISTA' or 'FISTA'.".format(algorithm))
            sys.exit(-1)

    elif penalty.lower() == 'l21':
        bound = tools.l21_norm_bound

        # Check minimization algorithm
        if algorithm.lower() == 'ista':
            minimizer = l21_norm_minimization
        elif algorithm.lower() == 'fista':
            minimizer = accelerated_l21_norm_minimization
        else:
            print("Minimization strategy {} not understood, "
                  "it must be 'ISTA' or 'FISTA'.".format(algorithm))
            sys.exit(-1)

    elif penalty.lower() == 'l21_lfro':
        print("Multi-task Elastic-net not yet implemented.")
        sys.exit(-1)
    else:
        print("Penalty {} not understood, "
              "it must be in ['trace', 'l21', 'l21_lfro'].".format(penalty))
        sys.exit(-1)

    return minimizer, bound


def kf_worker(minimizer, X_tr, Y_tr, tau_range, loss, tr_idx, vld_idx, i,
              results):
    """Worker for parallel KFold implementation."""
    Ws, _, _ = regularization_path(minimizer, X_tr, Y_tr,
                                   tau_range, loss)
    results[i] = {'W': Ws, 'tr_idx': tr_idx, 'vld_idx': vld_idx}


def model_selection(data, labels, tau_range, algorithm='FISTA', loss='square',
                    penalty='trace', cv_split=5):
    """Select the best tau in the range and return the best model.

    Parameters
    ----------
    data : (n, d) float ndarray
        training data matrix
    labels : (n, T) float ndarray
        training labels matrix
    tau_range : (n_tau, ) float ndarray
        range of regularization parameters (max_tau scaling factors)
    algorithm : string in {'ISTA', 'FISTA'}
        the minimization algorithm of choice, this could be either
        'ISTA' (default) or 'FISTA'
    loss : string in {'square', 'logit'}
        the selected loss function, this could be either 'square' or 'logit'
        ('logit' not yet implemeted)
    penalty : string in {'trace', 'l21', 'l21_lfro'}
        the penalty to be used, this could be 'trace' for
        nuclear-norm-penalized problems, 'l21' for multi-task lasso and
        'l21_lfro' for multi-task elastic-net.
    cv_split : int (optional, default=5)
        the number of K-fold cross-validation split used to perform parameter
        selection

    Returns
    -------
    out : dictionary
        out keys are:
            tau_range : (n_tau,) float ndarray
                the actual tau rage, i.e. the input one scaled for max_tau
            opt_tau : float
                the optimal tau value
            tr_err : (n_tau,) float ndarray
                the average CV training error for each tau
            vld_err
                the average CV validation error for each tau
            W_hat
                the estimated wheights matrix (model)
    """
    # Check minimization algorithm and penalty
    minimizer, bound = get_minimizer(algorithm, penalty)

    # Problem size
    n, d = data.shape

    # Get the maximum tau value
    max_tau = bound(data, labels)
    scaled_tau_range = tau_range * max_tau

    # Kfold starts here
    kf = KFold(n=n, n_folds=cv_split)
    jobs = []  # multiprocessing job list
    results = mp.Manager().dict()  # dictionary shared among processess

    # Submit each kfold job
    for i, (tr_idx, vld_idx) in enumerate(kf):
        X_tr = data[tr_idx, :]
        Y_tr = labels[tr_idx, :]
        X_vld = data[vld_idx, :]
        Y_vld = labels[vld_idx, :]

        p = mp.Process(target=kf_worker, args=(minimizer, X_tr,
                                               Y_tr, tau_range * max_tau, loss,
                                               tr_idx, vld_idx,
                                               i, results))
        jobs.append(p)
        p.start()

    # Collect the results
    for p in jobs:
        p.join()

    # Evaluate the errors
    tr_errors = np.zeros((cv_split, len(tau_range)))
    vld_errors = np.zeros((cv_split, len(tau_range)))
    for i in results.keys():
        Ws = results[i]['W']
        tr_idx = results[i]['tr_idx']
        vld_idx = results[i]['vld_idx']
        X_tr = data[tr_idx, :]
        Y_tr = labels[tr_idx, :]
        X_vld = data[vld_idx, :]
        Y_vld = labels[vld_idx, :]
        for j, W in enumerate(Ws):
            Y_pred_tr = np.dot(X_tr, W)
            Y_pred_vld = np.dot(X_vld, W)
            vld_errors[i, j] = (np.linalg.norm((Y_vld - Y_pred_vld),
                                               ord='fro')**2)/Y_vld.shape[0]
            tr_errors[i, j] = (np.linalg.norm((Y_tr - Y_pred_tr),
                                              ord='fro')**2)/Y_tr.shape[0]

    # Once all the training is done, get the best tau
    avg_vld_err = np.mean(vld_errors, axis=0)
    avg_tr_err = np.mean(tr_errors, axis=0)
    opt_tau = scaled_tau_range[np.argmin(avg_vld_err)]

    # Refit and return the best model
    W_hat, _ = minimizer(data, labels, opt_tau)

    # Output container
    out = {'tau_range': scaled_tau_range,
           'opt_tau': opt_tau,
           'avg_tr_err': avg_tr_err,
           'avg_vld_err': avg_vld_err,
           'tr_err': tr_errors,
           'vld_err': vld_errors,
           'W_hat': W_hat}
    return out
