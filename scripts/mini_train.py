#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal training script.

This script performs trace-norm penalized vector-valued regression (VVR) on a
given input dataset.
"""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import imp
import shutil
import argparse
import os
import cPickle as pkl
import seaborn as sns
import minimal as mini


def main(config_file):
    """Import configuration file and run VVR."""
    # Load the configuration file
    config_path = os.path.abspath(config_file)
    config = imp.load_source('mini_config', config_path)

    # Extract the needed information from config
    data = config.X  # (n, d) data matrix
    labels = config.Y  # (d, T) labels matrix
    tau_range = config.tau_range
    minimization = config.minimization_algorithm
    cv_split = config.cross_validation_split

    print("-------------- Minimal --------------")
    print("* Data matrix:\t\t   {} x {}".format(*data.shape))
    print("* Labels matrix:\t   {} x {}".format(*labels.shape))
    print("* Minimization algorithm:  {}".format(minimization))
    print("* Number of tau:\t   {}".format(len(tau_range)))
    print("* Cross-validation splits: {}".format(cv_split))

    out = mini.core.model_selection(data=data, labels=labels,
                                    tau_range=tau_range,
                                    algorithm=minimization,
                                    cv_split=cv_split)
    # Dump results
    root = config.output_root_folder
    folder = os.path.join(root, '_'.join(('mini', config.exp_tag,
                                          mini.extra.get_time())))
    filename = os.path.join(folder, 'results.pkl')
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(filename, 'w') as f:
        pkl.dump(out, f)
    print("* Results dumped in {}".format(filename))

    # Save simple cross-validation error plots
    filename = os.path.join(folder, 'cv-errors.pdf')
    sns.plt.semilogx(out['tau_range'], out['tr_err'], label='tr error')
    sns.plt.semilogx(out['tau_range'], out['vld_err'], label='vld error')
    sns.plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
    sns.plt.title("{}-Fold cross-validation error".format(cv_split))
    sns.plt.xlabel(r"$log_{10}(\tau)$")
    sns.plt.ylabel(r"$||Y_{vld} - Y_{pred}||_F$")
    sns.plt.savefig(filename)
    print("* Plot generated in {}".format(filename))
    print("-------------------------------------")


######################################################################

if __name__ == '__main__':
    __version__ = mini.__version__
    parser = argparse.ArgumentParser(description='Minimal script for '
                                                 'trace-norm penalized'
                                                 'vector-valued regression.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument("-c", "--create", dest="create", action="store_true",
                        help="create config file", default=False)
    parser.add_argument("configuration_file", help="specify config file",
                        default='mini_config.py')
    args = parser.parse_args()

    if args.create:
        std_config_path = os.path.join(mini.__path__[0], 'mini_config.py')
        # Check for .pyc
        if std_config_path.endswith('.pyc'):
            std_config_path = std_config_path[:-1]
        # Check if the file already exists
        if os.path.exists(args.configuration_file):
            parser.error("Minimal configuration file already exists")
        # Copy the config file
        shutil.copy(std_config_path, args.configuration_file)
    else:
        main(args.configuration_file)
