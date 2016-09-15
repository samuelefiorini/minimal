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
from minimal.algorithms import trace_norm_path
from minimal.algorithms import trace_norm_minimization
from minimal.algorithms import accelerated_trace_norm_minimization
from sklearn.cross_validation import KFold, train_test_split
from minimal import tools


def kf_worker(minimizer, X_tr, Y_tr, tau_range, tr_idx, vld_idx, i, results):
    """Worker for parallel KFold implementation."""
    Ws, _, _ = trace_norm_path(minimizer, X_tr, Y_tr,
                               tau_range, loss='square')
    results[i] = {'W': Ws, 'tr_idx': tr_idx, 'vld_idx': vld_idx}


def model_selection(data, labels, tau_range, algorithm='ISTA', cv_split=5):
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
    # Check minimization algorithm
    if algorithm == 'ISTA':
        minimizer = trace_norm_minimization
    elif algorithm == 'FISTA':
        minimizer = accelerated_trace_norm_minimization
    else:
        print("Minimization strategy {} not understood,"
              "it must be ISTA or FISTA.")
        sys.exit(-1)

    # Problem size
    n, d = data.shape

    # Get the maximum tau value
    max_tau = tools.trace_norm_bound(data, labels)
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
                                               Y_tr, tau_range * max_tau,
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
                                               ord='fro') ** 2) / len(Y_vld)
            tr_errors[i, j] = (np.linalg.norm((Y_tr - Y_pred_tr),
                                              ord='fro') ** 2) / len(Y_tr)

    # Once all the training is done, get the best tau
    avg_vld_err = np.mean(vld_errors, axis=0)
    avg_tr_err = np.mean(tr_errors, axis=0)
    opt_tau = scaled_tau_range[np.argmin(avg_vld_err)]

    # Refit and return the best model
    W_hat = minimizer(data, labels, opt_tau)

    # Output container
    out = {'tau_range': scaled_tau_range,
           'opt_tau': opt_tau,
           'tr_err': avg_tr_err,
           'vld_err': avg_vld_err,
           'W_hat': W_hat}
    return out
