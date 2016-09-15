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
import minimal as mini


def main(root):
    """Import trained model and test Minimal."""
    # Load test set

    # Load trained model
    res_file = os.path.join(os.path.abspath(root), filename[0])
    with open(res_file, 'r') as f:
        results = pkl.load(f)

    print results



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
