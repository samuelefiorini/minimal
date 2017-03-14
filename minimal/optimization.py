#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Internal algorithms implementations for penalized minimization problems.

This module contains implementations of proximal gradient methods such as
ISTA [1] and FISTA [1].

References
----------
[1] Beck, Amir, and Marc Teboulle. "A fast iterative shrinkage-thresholding
algorithm for linear inverse problems." SIAM journal on imaging sciences
2.1 (2009): 183-202.
"""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

__all__ = ['ISTA', 'FISTA']


def ISTA():
    """Iterative-Shrinking Thresholding Algorithm.

    Compute the solution of the minimization problem

                min loss(Y, XW) + tau pen(W)
                 W

    via forward-backward splitting.

    The samples are stored row-wise in X, W is the weight matrix to
    learn and each column of Y correspond to a single task (for vector-valued
    regression) or a binary entry (for multi-category classification).
    """
    pass


def FISTA():
    """Fast Iterative-Shrinking Thresholding Algorithm.

    Compute the solution of the minimization problem

                min loss(Y, XW) + tau pen(W)
                 W

    via acceperated forward-backward splitting.

    The samples are stored row-wise in X, W is the weight matrix to
    learn and each column of Y correspond to a single task (for vector-valued
    regression) or a binary entry (for multi-category classification).
    """
    pass
