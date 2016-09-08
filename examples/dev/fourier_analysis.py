#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyzing SVD and Fourier analysis relationship."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import numpy as np
import matplotlib.pyplot as plt
from SDG4ML.core.wrappers import generate_data
from minimal import algorithms as algo

def main():
    kwargs = {'n': 1000, 'd': 150, 'T': 20,
              'normalized': False, 'seed': seed}
    X, Y, W = generate_data(strategy='multitask', **kwargs)
    max_tau = trace_norm_bound(X, Y, loss='square')

    FX = np.fft.fft2(X)
    FX = np.fft.fftshift(FX)

    rFX = np.fft.fft2(algo.trace_norm_prox(X), max_tau)
    rFX = np.fft.fftshift(rFX)

    plt.figure()

    plt.subplot(131)
    plt.imshow(FX), plt.title('FX'), plt.colorbar()




if __name__ == '__main__':
    main()
