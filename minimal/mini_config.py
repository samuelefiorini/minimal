#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration file for Minimal."""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

from minimal import data_source

# --------------------------  EXPERMIENT INFO ------------------------- #
exp_tag = 'minimal_dev'
output_root_folder = 'results'
plotting_context = 'notebook'  # one of {paper, notebook, talk, poster}
file_format = 'pdf'  # or 'png'

# ----------------------------  INPUT DATA ---------------------------- #
data_file = 'data.csv'
labels_file = 'labels.csv'  # OPTIONAL
samples_on = 'rows'  # if samples lie on columns use 'cols' or 'col'
data_sep = ','  # the data separator. e.g., ',', '\t', ' ', ...
X, y, feat_names, index = data_source.load('custom',
                                           data_file, labels_file,
                                           samples_on=samples_on,
                                           sep=data_sep)
