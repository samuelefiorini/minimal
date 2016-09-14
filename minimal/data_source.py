#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Forked from ADENINE and SDG4ML.

Lite version of adenine.utils.data_source() function that inherits the
synthetic VVR problem generation from SDG4ML.core.strategies.multitask().
"""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Federico Tomasi, Annalisa Barla
#
# FreeBSD License
######################################################################

import sys
import numpy as np
import pandas as pd
import logging
from sklearn import datasets


def load_custom(x_filename, y_filename, samples_on='rows', **kwargs):
    """Load a custom dataset.

    This function loads the data matrix and the label vector returning a
    unique sklearn-like object dataSetObj.

    Parameters
    -----------
    x_filename : string
        The data matrix file name.

    y_filename : string
        The label vector file name.

    samples_on : string
        This can be either in ['row', 'rows'] if the samples lie on the row of
        the input data matrix, or viceversa in ['col', 'cols'] the other way
        around.

    kwargs : dict
        Arguments of pandas.read_csv function.

    Returns
    -----------
    data : sklearn.datasets.base.Bunch
        An instance of the sklearn.datasets.base.Bunch class, the meaningful
        attributes are .data, the data matrix, and .target, the label vector.
    """
    if x_filename is None:
        raise IOError("Filename for X must be specified with mode 'custom'.")

    if x_filename.endswith('.npy'):  # it an .npy file is provided
        try:  # labels are not mandatory
            y = np.load(y_filename)
        except IOError as e:
            y = None
            e.strerror = "No labels file provided"
            logging.error("I/O error({0}): {1}".format(e.errno, e.strerror))
        X = np.load(x_filename)
        if samples_on not in ['row', 'rows']:
            # data matrix must be n_samples x n_features
            X = X.T
        return datasets.base.Bunch(data=X, target=y,
                                   index=np.arange(X.shape[0]))

    elif x_filename.endswith('.csv') or x_filename.endswith('.txt'):
        y = None
        kwargs.setdefault('header', 0)  # header on first row
        kwargs.setdefault('index_col', 0)  # indexes on first
        try:
            dfx = pd.read_csv(x_filename, **kwargs)
            if samples_on not in ['row', 'rows']:
                # data matrix must be n_samples x n_features
                dfx = dfx.transpose()
            if y_filename is not None:
                # Before loading labels, remove parameters that were likely
                # specified for data only.
                kwargs.pop('usecols', None)
                y = pd.read_csv(y_filename, **kwargs).as_matrix().ravel()

        except IOError as e:
            e.strerror = "Can't open {} or {}".format(x_filename, y_filename)
            logging.error("I/O error({0}): {1}".format(e.errno, e.strerror))
            sys.exit(-1)

        return datasets.base.Bunch(data=dfx.as_matrix(),
                                   feature_names=dfx.columns.tolist(),
                                   target=y, index=dfx.index.tolist())


def make_multitask(n=100, d=150, T=1, amplitude=3.5, sigma=1,
                   normalized=False, seed=None, **kwargs):
    """Generate a multitask vector-valued regression (X,Y) problem.

    The relationship between input and output is given by:

                                            Y = X*W + noise

    where X ~ N(0, 1), W ~ U(-amplitude, +amplitude), noise ~ N(0,sigma).

    Parameters
    ----------
    n : int, optional (default is `100`)
        number of samples
    d : int, optional (default is `150`)
        total number of dimensions
    T : int, optional (default is `1`)
        number of tasks
    amplitude : float,  optional (default is `3.5`)
        amplitude of the generative linear model
    sigma : float, optional (default is `1`)
        Gaussian noise std
    normalized : bool, optional (default is `False`)
        if normalized is true than the data matrix is normalized as
        data/sqrt(n)
    seed : float, optional (default is `None`)
        random seed initialization

    Returns
    -------
    X : (n, d) ndarray
        data matrix
    Y : (n, T) ndarray
        label vector
    beta : (d, T) ndarray
        real beta vector
    """
    if seed is not None:
        state0 = np.random.get_state()
        np.random.seed(seed)

    if normalized:
        factor = np.sqrt(n)
    else:
        factor = 1

    X = np.random.randn(n, d)/factor
    W = np.random.uniform(low=-np.abs(amplitude),
                          high=+np.abs(amplitude),
                          size=(d, T))
    Y = np.dot(X, W) + sigma * np.random.randn(n, T)

    if seed is not None:  # restore random seed
        np.random.set_state(state0)

    # Crate dummy names for variables and samples
    _feats = ['feat_'+str(i) for i in range(X.shape[1])]
    _indexes = ['sample_'+str(i) for i in range(X.shape[0])]

    return datasets.base.Bunch(data=X,
                               feature_names=_feats,
                               target=Y, index=_indexes, W_star=W)


def load(opt='custom', x_filename=None, y_filename=None, n_samples=0,
         samples_on='rows', **kwargs):
    """Load a specified dataset.

    This function can be used either to load one of the standard scikit-learn
    datasets or a different dataset saved as X.npy Y.npy in the working
    directory.

    Parameters
    -----------
    opt : {'iris', 'digits', 'diabetes', 'boston', 'circles', 'moons',
          'custom'}, default: 'custom'
        Name of a predefined dataset to be loaded.

    x_filename : string, default : None
        The data matrix file name.

    y_filename : string, default : None
        The label vector file name.

    n_samples : int
        The number of samples to be loaded. This comes handy when dealing with
        large datasets. When n_samples is less than the actual size of the
        dataset this function performs a random subsampling that is stratified
        w.r.t. the labels (if provided).

    samples_on : string
        This can be either in ['row', 'rows'] if the samples lie on the row of
        the input data matrix, or viceversa in ['col', 'cols'] the other way
        around.

    data_sep : string
        The data separator. For instance comma, tab, blank space, etc.

    Returns
    -----------
    X : array of float, shape : n_samples x n_features
        The input data matrix.

    y : array of float, shape : n_samples
        The label vector; np.nan if missing.

    feature_names : array of integers (or strings), shape : n_features
        The feature names; a range of number if missing.

    index : list of integers (or strings)
        This is the samples identifier, if provided as first column (or row) of
        of the input file. Otherwise it is just an incremental range of size
        n_samples.
    """
    data = None
    try:
        if opt.lower() == 'custom':
            data = load_custom(x_filename, y_filename, samples_on, **kwargs)
        elif opt.lower() in ['synthetic', 'vvr']:
            data = make_multitask(n=n_samples, **kwargs)
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))

    X, y = data.data, data.target

    feat_names = data.feature_names if hasattr(data, 'feature_names') \
        else np.arange(X.shape[1])
    index = np.array(data.index) if hasattr(data, 'index') \
        else np.arange(X.shape[0])

    if hasattr(data, 'W_star'):
        return X, y, feat_names, index, data.W_star
    else:
        return X, y, feat_names, index
