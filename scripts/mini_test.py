#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal testing script.

This script, once minimal is trained on an appropriate training set, evaluates
the error on a test set.
"""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import imp
import sys
import os
import argparse
import cPickle as pkl
import numpy as np
import minimal as mini


def main(root):
    """Import trained model and test Minimal."""
    # Load the configuration file
    config_path = os.path.abspath(os.path.join(root, 'mini_config.py'))
    config = imp.load_source('mini_config', config_path)

    # Extract the needed information from config
    test_data = config.test_X  # (n, d) test data matrix
    test_labels = config.test_Y  # (d, T) test labels matrix
    cv_split = config.cross_validation_split

    print("-------------- Minimal test ------------------")
    print("* Data matrix:\t\t   {} x {}".format(*test_data.shape))
    print("* Labels matrix:\t   {} x {}".format(*test_labels.shape))

    # Load trained model
    res_file = os.path.join(os.path.abspath(root), 'results.pkl')
    with open(res_file, 'r') as f:
        results = pkl.load(f)

    # Get the weights
    W_hat = results['W_hat']

    # Predict the labels of the test set
    pred_labels = np.dot(test_data, W_hat)
    pred_error = (np.linalg.norm((test_labels - pred_labels),
                  ord='fro') ** 2) / test_labels.shape[0]

    print("* Prediction error:\t   {:.4}".format(pred_error))
    filename = os.path.join(root, 'cv-errors')
    mini.plotting.errors(results=results, cv_split=cv_split, filename=filename,
                         test_error=pred_error,
                         file_format=config.file_format,
                         context=config.plotting_context)
    print("----------------------------------------------")


######################################################################

if __name__ == '__main__':
    __version__ = mini.__version__
    parser = argparse.ArgumentParser(description='Minimal script for '
                                                 'model testing.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s v' + __version__)
    parser.add_argument("result_folder", help="specify results directory")
    args = parser.parse_args()
    root_folder = args.result_folder
    filename = [f for f in os.listdir(root_folder)
                if os.path.isfile(os.path.join(root_folder, f)) and
                f.endswith('.pkl')]
    if not filename:
        sys.stderr.write("No .pkl file found in {}. Aborting...\n"
                         .format(root_folder))
        sys.exit(-1)

    # Run analysis
    main(root_folder)
