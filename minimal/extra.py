#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test ISTA/FISTA strategies for nuclear norm minimization."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import sys
import time
from datetime import datetime


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


def sec_to_time(seconds):
    """Transform seconds into a formatted time string.

    Parameters
    -----------
    seconds : int
        Seconds to be transformed.

    Returns
    -----------
    time : string
        A well formatted time string.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def get_time():
    """Handy time flat creator."""
    return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H.%M.%S')


def timed(function):
    """Decorator that measures wall time of the decored function."""
    def timed_function(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        print("\n{} - Elapsed time : {} s\n"
              .format(function.__name__, sec_to_time(time.time() - t0)))
        return result
    return timed_function
