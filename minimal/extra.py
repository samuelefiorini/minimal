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
    """Decorator that tests the decored function."""
    def tested_function(*args, **kwargs):

        s = 0
        while True:
            try:
                print("\nTesting strategies with seed: {}\n".format(s))
                function(*args, seed=s, **kwargs)
                s += 1
            except:
                e = sys.exc_info()[0]
                print("ERROR: {}".format(e))
                print("Seed: {}".format(s))

        # t0 = time.time()
        # result = function(*args, **kwargs)
        # print("\nAdenine {} - Elapsed time : {} s\n"
        #       .format(function.__name__, sec_to_time(time.time() - t0)))
        # return result

    return tested_function
