#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example of trace-norm minimazion problem."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import numpy as np
from minimal.algorithms import trace_norm_minimization
from SDG4ML.core.wrappers import generate_data
from sklearn.cross_validation import train_test_split


def main():
    """Solve a synthetic vector-valued regression problem."""
    kwargs = {'n': 100, 'd': 300, 'k': 15, 'T': 5,
              'amplitude': 3.5, 'normalized': False, 'seed': 42}

    X, Y, W = generate_data(strategy='multitask', **kwargs)
    Xtr, Xts, Ytr, Yts = train_test_split(X, Y, test_size=0.33,
                                          random_state=kwargs['seed'])
    W_hat = trace_norm_minimization(Xtr, Ytr)
    Y_pred = np.dot(Xts, W_hat)
    Y_pred_tr = np.dot(Xtr, W_hat)
    ts_err = np.linalg.norm((Yts - Y_pred), ord='fro')
    tr_err = np.linalg.norm((Ytr - Y_pred_tr), ord='fro')
    W_err = np.linalg.norm((W - W_hat), ord='fro')

    print("Test error: {}".format(ts_err))
    print("Train error: {}".format(tr_err))
    print("Recontruction error: {}".format(W_err))


if __name__ == '__main__':
    main()
