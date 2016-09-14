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
import numpy as np


def model_selection(data, labels, tau_range, algorithm='ISTA', cv_split=5):
    """asdsdsas.

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
    best_model : (d, T) float ndarray
        the best model return after refitting the algorithm of choice on the
        whole dataset using the best parameter
    errors : list of floats
        a list of performance metrics [## FILL HERE ##]
    """
    pass
