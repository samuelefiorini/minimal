#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal plotting utilities.

This module contains the functions to generate intuitive plots for results
intepretation.
"""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import seaborn as sns
import numpy as np
import pandas as pd


def lite_errors(results, cv_split, filename, file_format='png',
                context='notebook'):
    """Plot training/validation error curves.

    Parameters
    -----------
    results : dictionary
        output of minimal.core.model_selection
    """
    sns.plt.semilogx(results['tau_range'], results['avg_tr_err'],
                     '-o', label='tr error')
    sns.plt.semilogx(results['tau_range'], results['avg_vld_err'],
                     '-o', label='vld error')
    sns.plt.semilogx(results['opt_tau'], min(results['avg_vld_err']),
                     'h', label=r'opt $\tau$', c='#a40000')
    sns.plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
    sns.plt.title("{}-Fold cross-validation error".format(cv_split))
    sns.plt.xlabel(r"$log_{10}(\tau)$")
    sns.plt.ylabel(r"$\frac{1}{n}||Y - Y_{pred}||_F^2$")
    sns.plt.savefig(filename+'_lite_.'+file_format)


def errors(results, cv_split, filename, test_error=None,
           file_format='png', context='notebook'):
    """Plot training/validation error curves.

    This function follows the tutorial available at:
    https://stanford.edu/~mwaskom/software/seaborn/examples/timeseries_from_dataframe.html

    Parameters
    -----------
    results : dictionary
        output of minimal.core.model_selection
    cv_split : int (optional, default=5)
        the number of K-fold cross-validation split used to perform parameter
        selection
    filename : string
        the output filename
    test_error : float (optional, default=None)
        when not none, a the test error is represented as dashed
        horizontal line
    file_format : string
        this could be 'png', 'pdf' and so on
    context : string (optional, default 'notebook')
        the seaborn plotting context
    """
    sns.plt.clf()
    sns.set_context(context)

    _tau_range = [results['tau_range'].tolist() * cv_split]
    _tau_range = [_tau_range * 2]
    _tau_range = np.array(_tau_range).ravel()  # gammas timepoint
    tr_err = results['tr_err'].ravel()
    vld_err = results['vld_err'].ravel()
    errors = np.hstack((tr_err, vld_err))  # gammas value
    tr_unit = np.array([[i]*len(results['tau_range'])
                       for i in range(cv_split)]).ravel()
    vld_unit = np.array([[i]*len(results['tau_range'])
                        for i in range(cv_split)]).ravel()
    unit = np.hstack((tr_unit, vld_unit))  # gammas subject
    condition = np.array([[i]*len(tr_unit) for i in
                         ['tr error', 'vld error']]).ravel()  # gammas ROI

    # DataFrame for seaborn plot
    gammas = pd.DataFrame()
    gammas['xaxis'] = _tau_range
    gammas['errors'] = errors
    gammas['unit'] = unit
    gammas['condition'] = condition

    # Plot the response with standard error
    sns.tsplot(data=gammas, time="xaxis", unit="unit",
               condition="condition", value="errors")
    sns.plt.semilogx(results['opt_tau'], min(results['avg_vld_err']),
                     'h', label=r'opt $\tau$', c='#a40000')

    # Check if the test error should be represented or not
    if test_error is not None:
        sns.plt.axhline(test_error, linestyle='dashed', label='test error',
                        color=sns.xkcd_rgb['dark brown'])

    sns.plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
    sns.plt.title("{}-Fold cross-validation error".format(cv_split))
    sns.plt.xlabel(r"$log_{10}(\tau)$")
    sns.plt.ylabel(r"$\frac{1}{n}||Y - Y_{pred}||_F^2$")

    sns.plt.savefig(filename+'.'+file_format)
