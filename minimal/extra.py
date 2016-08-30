#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test ISTA/FISTA strategies for nuclear norm minimization."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import sys


def test(function):
    """Testing decorator."""
    def tested_function(*args, **kwargs):
        s = 0
        while True:
            try:
                print("\nTesting strategies with seed: {}\n".format(s))
                function(*args, seed=s, **kwargs)
                s += 1
            except (KeyboardInterrupt, SystemExit):
                print("Exit signal received.")
                break
            except:
                e = sys.exc_info()[0]
                print("ERROR: {}".format(e))
                print("Seed: {}".format(s))
                break
    return tested_function
