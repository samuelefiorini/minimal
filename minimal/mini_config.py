#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration file for Minimal."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

import numpy as np
from minimal import data_source

# -------------------------- CONTEXT INFO ---------------------------- #
exp_tag = 'dev'
output_root_folder = 'results'
plotting_context = 'notebook'  # one of {paper, notebook, talk, poster}
file_format = 'png'  # or 'pdf'

# ---------------------------- INPUT DATA ---------------------------- #
# data_file = 'data.csv'
# labels_file = 'labels.csv'  # OPTIONAL
# samples_on = 'rows'  # if samples lie on columns use 'cols' or 'col'
# data_sep = ','  # the data separator. e.g., ',', '\t', ' ', ...
# X, y, feat_names, index = data_source.load('custom',
#                                            data_file, labels_file,
#                                            samples_on=samples_on,
#                                            sep=data_sep)
X, Y, feat_names, index, W = data_source.load(opt='synthetic',
                                              n_samples=20,
                                              d=7, T=5, sigma=5)
# X, Y, feat_names, index, W = data_source.load(opt='synthetic',
#                                               n_samples=1300,
#                                               d=150, T=30, sigma=5)


# ---------------------------- EXPERIMENT SETTING -------------------- #
minimization_algorithm = 'FISTA'  # in ['ISTA', 'FISTA']
cross_validation_split = 5  # number of Kfold CV split for parameter selection
tau_range = np.logspace(-3, 0, 20)  # scaling factors for TAU_MAX
