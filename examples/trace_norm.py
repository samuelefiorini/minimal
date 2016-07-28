#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example of trace-norm minimazion problem."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import numpy as np
import matplotlib.pyplot as plt
from minimal.algorithms import trace_norm_minimization
from SDG4ML.core.wrappers import generate_data
from sklearn.cross_validation import train_test_split


def main():
    """Solve a synthetic vector-valued regression problem."""

    # The data generation parameter(s)
    kwargs = {'n': 12, 'd': 7, 'k': 3, 'T': 5,
              'amplitude': 3.5, 'normalized': False, 'seed': None}

    X, Y, W = generate_data(strategy='multitask', **kwargs)
    Xtr, Xts, Ytr, Yts = train_test_split(X, Y, test_size=0.33,
                                          random_state=kwargs['seed'])

    print("W size: {} x {}".format(*W.shape))

    # The learning parameter(s)
    tau = 1

    W_hat, objs = trace_norm_minimization(Xtr, Ytr, tau)
    Y_pred = np.dot(Xts, W_hat)
    Y_pred_tr = np.dot(Xtr, W_hat)
    ts_err = np.linalg.norm((Yts - Y_pred), ord='fro')
    tr_err = np.linalg.norm((Ytr - Y_pred_tr), ord='fro')
    W_err = np.linalg.norm((W - W_hat), ord='fro')

    print("Test error: {}".format(ts_err))
    print("Train error: {}".format(tr_err))
    print("Recontruction error: {}".format(W_err))

    print("-----------------------------------------")
    print("Real W:")
    print(W)
    print("-----------------------------------------")
    print("Estimated W:")
    print(W_hat)

    plt.plot(np.arange(len(objs)), objs, '-o')
    plt.xlabel('iterations')
    plt.ylabel('objective function')
    plt.show()


if __name__ == '__main__':
    main()
