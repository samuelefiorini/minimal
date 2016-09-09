#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example of trace-norm minimazion problem."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import numpy as np
import seaborn as sns
from minimal.algorithms import trace_norm_minimization
from minimal.algorithms import accelerated_trace_norm_minimization
from minimal.tools import trace_norm_bound
from minimal.tools import objective_function
from minimal.extra import test
from SDG4ML.core.wrappers import generate_data
from sklearn.cross_validation import train_test_split


def single_run(minimization, Xtr, Xts, Ytr, Yts, tau, W, plot=False):
    """Single run of the minimzation algorithm."""
    W_hat, objs, iters = minimization(Xtr, Ytr, tau,
                                      return_iter=True, tol=1e-7)

    Y_pred = np.dot(Xts, W_hat)
    Y_pred_tr = np.dot(Xtr, W_hat)
    ts_err = np.linalg.norm((Yts - Y_pred), ord='fro')
    tr_err = np.linalg.norm((Ytr - Y_pred_tr), ord='fro')
    W_err = np.linalg.norm((W - W_hat), ord='fro')

    print("-----------------------------------------")
    print("tau : {}".format(tau))
    print("Test error: {}".format(ts_err))
    print("Train error: {}".format(tr_err))
    print("Recontruction error: {}".format(W_err))
    print("Iters : {}".format(iters))

    if plot:
        sns.plt.plot(np.arange(len(objs)), objs, '-o')
        sns.plt.xlabel('iterations')
        sns.plt.ylabel('objective function')
        sns.plt.show()

    return tr_err, ts_err, W_err, objs, iters


# @test
def main(seed=None, **kwargs):
    """Solve a synthetic vector-valued regression problem."""
    # The data generation parameter(s)
    # kwargs = {'n': 12, 'd': 7, 'T': 5,
    #           'normalized': False, 'seed': seed}
    kwargs = {'n': 100, 'd': 50, 'T': 20, 'sigma': 5,
              'normalized': False, 'seed': seed}

    X, Y, W = generate_data(strategy='multitask', **kwargs)
    Xtr, Xts, Ytr, Yts = train_test_split(X, Y, test_size=0.33,
                                          random_state=kwargs['seed'])
    # Problem parameters
    loss = 'square'
    # Objective function value for W
    objW = objective_function(Xtr, Ytr, W, loss=loss)

    # The learning parameter(s)
    max_tau = trace_norm_bound(Xtr, Ytr, loss=loss)
    tau_range = np.logspace(-4, 0, 20)

    # The minimizer of choiche
    minimizers = [trace_norm_minimization,
                  accelerated_trace_norm_minimization][::-1]
    names = ['ISTA', 'FISTA'][::-1]
    for minimizer, name in zip(minimizers, names):
        print("*** {} ***".format(name))

        # Output containers
        tr_err_list = list()
        ts_err_list = list()
        W_err_list = list()
        objs_list = list()
        iters_list = list()
        for t in tau_range:
            tau = max_tau * t
            # print("tau = {}".format(tau))
            tr_err, ts_err, W_err, obj, iters = single_run(minimizer,
                                                           Xtr, Xts,
                                                           Ytr, Yts,
                                                           tau, W)
            tr_err_list.append(tr_err)
            ts_err_list.append(ts_err)
            # the last value is the one for which the
            # algorithm has reached convergence
            objs_list.append(obj)
            iters_list.append(iters)
            W_err_list.append(W_err)

        print("***********************************************\n")

        opt_tau = tau_range[np.argmin(ts_err_list)] * max_tau
        print("Best tau: {}\n".format(opt_tau))

        # Plot section
        sns.set_context("notebook")
        sns.plt.figure()
        sns.plt.subplot(221)
        sns.plt.semilogx(tau_range * max_tau, tr_err_list, '-o',
                         label='train error')
        sns.plt.semilogx(tau_range * max_tau, ts_err_list, '-o',
                         label='test error')
        sns.plt.semilogx(opt_tau, np.min(ts_err_list),
                         'h', label=r'opt $\tau$', c='#a40000')
        sns.plt.ylabel(r"$||Y - Y_{pred}||_F$")
        sns.plt.title("Tr/Ts Errors")
        sns.plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=2, mode="expand", borderaxespad=0.)

        sns.plt.subplot(222)
        sns.plt.title("Reconstruction errors")
        sns.plt.semilogx(tau_range * max_tau, W_err_list, '-o')
        sns.plt.semilogx(opt_tau, W_err_list[np.argmin(ts_err_list)],
                         'h', label=r'opt $\tau$', c='#a40000')
        sns.plt.ylabel(r"$||W - \hat{W}||_F$")

        sns.plt.subplot(223)
        sns.plt.semilogx(tau_range * max_tau,
                         np.array(objs_list) / objW, '-o')
        sns.plt.ylabel("Normalized obj fun at convergency")
        sns.plt.xlabel(r"$log_{10}(\tau)$")

        sns.plt.subplot(224)
        sns.plt.semilogx(tau_range * max_tau, iters_list, '-o')
        sns.plt.ylabel("Iters")
        sns.plt.xlabel(r"$log_{10}(\tau)$")
        sns.plt.suptitle(name)
        sns.plt.savefig(name+'.png')

if __name__ == '__main__':
    # main(seed=8)
    main()
