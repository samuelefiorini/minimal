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
from minimal import algorithms as al
from skimage import data
from skimage import color
from scipy import signal as sgnl

def show(A):
    plt.gray()
    plt.imshow(A)
    plt.show()

def main():
    # Load image
    X = color.rgb2gray(data.astronaut())

    # Prox norm
    TX = al.trace_norm_prox(X, alpha=10)

    # Fourier transform
    FX = np.fft.rfft2(X, s=X.shape)
    FTX = np.fft.rfft2(TX, s=TX.shape)

    # Frequency response
    H = np.fft.fftshift(FTX / FX)

    # Impulsive response
    h = np.fft.irfft2(H)

    #### TEST ####
    TX_hat = sgnl.fftconvolve(X, h)

    show(TX_hat)





if __name__ == '__main__':
    main()
