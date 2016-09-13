#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal main routine script.

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
import minimal as mini


def main(config_file):
    """Import configuration file and run VVR."""

    # Load the configuration file
    config_path = os.path.abspath(config_file)
    config = imp.load_source('mini_config', config_path)

    print config


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