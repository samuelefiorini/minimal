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
from sklearn.cross_validation import KFold
from minimal import tools


def kf_worker(minimizer, X_tr, Y_tr, tau_range, i, results):
    """Worker for parallel KFold implementation."""
    Ws, _, _ = trace_norm_path(minimizer, X_tr, Y_tr,
                               tau_range, loss='square')
    results[i] = Ws


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

    # Kfold starts here
    kf = KFold(n=n, n_folds=cv_split)
    tr_errors = np.empty((cv_split, len(tau_range)))
    vld_errors = np.empty((cv_split, len(tau_range)))
    jobs = []  # multiprocessing job list
    results = mp.Manager().dict()  # dictionary shared among processess

    # Submit each kfold job
    for i, (tr_idx, vld_idx) in enumerate(kf):
        X_tr = data[tr_idx, :]
        X_vld = data[vld_idx, :]
        Y_tr = labels[tr_idx, :]
        Y_vld = labels[vld_idx, :]

        p = mp.Process(target=kf_worker, args=(minimizer, X_tr,
                                               Y_tr, tau_range * max_tau,
                                               i, results))
        jobs.append(p)
        p.start()

    # Collect the results
    for p in jobs:
        p.join()

    # Evaluate the errors
    for i, w in enumerate(results.keys()):
        Ws = results[w]
        for j, W in enumerate(Ws):
            Y_pred_tr = np.dot(X_tr, W)
            Y_pred_vld = np.dot(X_vld, W)
            vld_errors[i, j] = np.linalg.norm((Y_vld - Y_pred_vld), ord='fro')
            tr_errors[i, j] = np.linalg.norm((Y_tr - Y_pred_tr), ord='fro')

    # Once all the training is done, get the best tau
    avg_vld_err = np.mean(vld_errors, axis=0)
    avg_tr_err = np.mean(tr_errors, axis=0)
    opt_tau = tau_range[np.argmin(avg_vld_err)] * max_tau

    # Refit and return the best model
    W_hat = minimizer(data, labels, opt_tau)

    # Output container
    out = {'tau_range': tau_range * max_tau,
           'opt_tau': opt_tau,
           'tr_err': avg_tr_err,
           'vld_err': avg_vld_err,
           'W_hat': W_hat}
    return out
