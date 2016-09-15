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


def errors(results, cv_split, filename):
    """Plot training/validation error curves.

    Parameters
    -----------
    results : dictionary
        output of minimal.core.model_selection
    """
    sns.plt.semilogx(results['tau_range'], results['tr_err'],
                     '-o', label='tr error')
    sns.plt.semilogx(results['tau_range'], results['vld_err'],
                     '-o', label='vld error')
    sns.plt.semilogx(results['opt_tau'], min(results['vld_err']),
                     'h', label=r'opt $\tau$', c='#a40000')
    sns.plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
    sns.plt.title("{}-Fold cross-validation error".format(cv_split))
    sns.plt.xlabel(r"$log_{10}(\tau)$")
    sns.plt.ylabel(r"$\frac{1}{n}||Y - Y_{pred}||_F^2$")
    sns.plt.savefig(filename)
