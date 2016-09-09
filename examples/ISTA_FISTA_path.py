#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example of trace-norm minimazion problem with warm restart."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import numpy as np
import seaborn as sns
from minimal.algorithms import trace_norm_minimization
from minimal.algorithms import accelerated_trace_norm_minimization
from minimal.algorithms import trace_norm_path
from minimal.tools import objective_function
from minimal.extra import test
from SDG4ML.core.wrappers import generate_data
from sklearn.cross_validation import train_test_split


def main():
    """Solve a synthetic vector-valued regression problem with warm restart."""

    # The data generation parameter(s)
    _kwargs = {'n': 12, 'd': 7, 'T': 5, 'sigma': 5,
               'normalized': False, 'seed': 10}
    # kwargs = {'n': 100, 'd': 50, 'T': 20, 'sigma': 5,
    #           'normalized': False, 'seed': seed}

    X, Y, W = generate_data(strategy='multitask', **_kwargs)
    Xtr, Xts, Ytr, Yts = train_test_split(X, Y, test_size=0.33,
                                          random_state=_kwargs['seed'])
    # Problem parameters
    loss = 'square'
    # Objective function value for W
    objW = objective_function(Xtr, Ytr, W, loss=loss)

    # The learning parameter(s)
    tau_range = np.logspace(-4, 0, 20)  # scaling factor

    # The minimizer of choiche
    minimizers = [trace_norm_minimization,
                  accelerated_trace_norm_minimization]
    names = ['ISTA', 'FISTA']
    for minimizer, name in zip(minimizers, names):
        print("*** {} ***".format(name))
        W_list, objs_list, iters_list = trace_norm_path(minimizer, Xtr, Ytr,
                                                        tau_range,
                                                        loss='square')

        # Evaluate training/test/recontruction errors
        tr_err_list = list()
        ts_err_list = list()
        W_err_list = list()

        for W_hat in W_list:
            Y_pred = np.dot(Xts, W_hat)
            Y_pred_tr = np.dot(Xtr, W_hat)
            ts_err = np.linalg.norm((Yts - Y_pred), ord='fro')
            tr_err = np.linalg.norm((Ytr - Y_pred_tr), ord='fro')
            W_err = np.linalg.norm((W - W_hat), ord='fro')

            # Save results into their containers
            tr_err_list.append(tr_err)
            ts_err_list.append(ts_err)
            W_err_list.append(W_err)

        # Identify the optimal scalinf factor for tau
        opt_tau = tau_range[np.argmin(ts_err_list)]
        print("Best tau: {} (scaling factor)\n".format(opt_tau))

        # Plot section
        sns.set_context("notebook")
        sns.plt.figure()
        sns.plt.subplot(221)
        sns.plt.semilogx(tau_range, tr_err_list, '-o',
                         label='train error')
        sns.plt.semilogx(tau_range, ts_err_list, '-o',
                         label='test error')
        sns.plt.semilogx(opt_tau, np.min(ts_err_list),
                         'h', label=r'opt $\tau$', c='#a40000')
        sns.plt.ylabel(r"$||Y - Y_{pred}||_F$")
        sns.plt.title("Tr/Ts Errors")
        sns.plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=2, mode="expand", borderaxespad=0.)

        sns.plt.subplot(222)
        sns.plt.title("Reconstruction errors")
        sns.plt.semilogx(tau_range, W_err_list, '-o')
        sns.plt.semilogx(opt_tau, W_err_list[np.argmin(ts_err_list)],
                         'h', label=r'opt $\tau$', c='#a40000')
        sns.plt.ylabel(r"$||W - \hat{W}||_F$")

        sns.plt.subplot(223)
        sns.plt.semilogx(tau_range,
                         np.array(objs_list) / objW, '-o')
        sns.plt.ylabel("Normalized obj fun at convergency")
        sns.plt.xlabel(r"$log_{10}(\tau)$")

        sns.plt.subplot(224)
        sns.plt.semilogx(tau_range, iters_list, '-o')
        sns.plt.ylabel("Iters")
        sns.plt.xlabel(r"$log_{10}(\tau)$")
        sns.plt.suptitle(name)
        sns.plt.savefig(name+'_warm.png')


if __name__ == '__main__':
    main()
