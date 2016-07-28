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
import seaborn
from minimal.algorithms import trace_norm_minimization
from SDG4ML.core.wrappers import generate_data
from sklearn.cross_validation import train_test_split


def single_run(Xtr, Xts, Ytr, Yts, tau, W, plot=False):
    """Single run of the minimzation algorithm."""
    W_hat, objs, iters = trace_norm_minimization(Xtr, Ytr,
                                                 tau, return_iter=True)
    Y_pred = np.dot(Xts, W_hat)
    Y_pred_tr = np.dot(Xtr, W_hat)
    ts_err = np.linalg.norm((Yts - Y_pred), ord='fro')
    tr_err = np.linalg.norm((Ytr - Y_pred_tr), ord='fro')
    W_err = np.linalg.norm((W - W_hat), ord='fro')

    print("Test error: {}".format(ts_err))
    print("Train error: {}".format(tr_err))
    print("Recontruction error: {}".format(W_err))
    print("Iters : {}".format(iters))

    print("-----------------------------------------")
    print("Estimated W:")
    print(W_hat)

    if plot:
        plt.plot(np.arange(len(objs)), objs, '-o')
        plt.xlabel('iterations')
        plt.ylabel('objective function')
        plt.show()

    return tr_err, ts_err, W_err, objs, iters


def main():
    """Solve a synthetic vector-valued regression problem."""

    # The data generation parameter(s)
    kwargs = {'n': 12, 'd': 7, 'k': 3, 'T': 5,
              'amplitude': 3.5, 'normalized': False, 'seed': None}

    X, Y, W = generate_data(strategy='multitask', **kwargs)
    Xtr, Xts, Ytr, Yts = train_test_split(X, Y, test_size=0.33,
                                          random_state=kwargs['seed'])
    print("-----------------------------------------")
    print("W size: {} x {}\n W = \n".format(*W.shape))
    print(W)

    # The learning parameter(s)
    max_tau = np.linalg.norm(np.dot(Xtr.T, Ytr), ord=2) * (2.0/Xtr.shape[0])  # the largest singular value
    tau_range = np.logspace(-4, 0, 20)

    tr_err_list = list()
    ts_err_list = list()
    W_err_list = list()
    objs_list = list()
    iters_list = list()
    for t in tau_range:
        tau = max_tau * t
        print("tau = {}".format(tau))
        tr_err, ts_err, W_err, objs, iters = single_run(Xtr, Xts,
                                                        Ytr, Yts, tau, W)

        tr_err_list.append(tr_err)
        ts_err_list.append(ts_err)
        # the last value is the one for which the
        # algorithm has reached convergence
        objs_list.append(objs[-1])
        iters_list.append(iters)
        W_err_list.append(W_err)

    print("***********************************************")

    opt_tau = tau_range[np.argmin(ts_err_list)]  * max_tau
    print("Best tau: {}".format(opt_tau))

    # Plot section
    plt.figure()
    plt.subplot(221)
    plt.semilogx(tau_range * max_tau, tr_err_list, '-o', label='train error')
    plt.semilogx(tau_range * max_tau, ts_err_list, '-o', label='test error')
    plt.semilogx(opt_tau, np.min(ts_err_list),
                 'h', label=r'opt $\tau$', c='#a40000')
    plt.ylabel(r"$||Y - Y_{pred}||_F$")
    plt.title("Tr/Ts Errors")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

    plt.subplot(222)
    plt.title("Reconstruction errors")
    plt.semilogx(tau_range * max_tau, W_err_list, '-o')
    plt.semilogx(opt_tau, W_err_list[np.argmin(ts_err_list)],
                 'h', label=r'opt $\tau$', c='#a40000')
    plt.ylabel(r"$||W - \hat{W}||_F$")

    plt.subplot(223)
    plt.semilogx(tau_range * max_tau, objs_list, '-o')
    plt.ylabel("Objective function at convergence")
    plt.xlabel(r"$log_{10}(\tau)$")

    plt.subplot(224)
    plt.semilogx(tau_range * max_tau, iters_list, '-o')
    plt.ylabel("Iters")
    plt.xlabel(r"$log_{10}(\tau)$")
    plt.show()


if __name__ == '__main__':
    main()
