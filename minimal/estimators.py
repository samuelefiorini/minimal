"""Minimal estimators.

This module contains scikit-learn compliant estimators for minimal methods.
"""

######################################################################
# Copyright (C) 2016 Samuele Fiorini, Annalisa Barla
#
# FreeBSD License
######################################################################

from functools import partial
from minimal.optimization import __algorithms__
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin
from sklearn.linear_model.base import _pre_fit, _preprocess_data
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import (check_array, check_X_y,
                           compute_sample_weight, column_or_1d)

import warnings
import numpy as np

class GroupLasso(LinearModel, RegressorMixin):
    """Linear regression with Group Lasso penalty as regularizer.

     Minimizes the objective function::

            (1 / n_samples) * ||y - Xw||^2_2 + alpha * GL(w)

    where ::

            GL(w) = ||( ||group_0||^2_2, ... ,||group_G||^2_2])||_2

    is the Group Lasso penalty.

    Parameters
    ----------
    alpha : float
        regularization parameter
    groups : list of lists (used only for group-lasso)
        the outer list represents the groups and the
        inner lists represent the variables in the groups. E.g. [[1, 2],
        [2, 3]] contains two groups ([1, 2] and [2, 3]) with variable 1 and
        2 in the first group and variables 2 and 3 in the second group.
    algorithm : string
        the selected minimization algorithm in {'ista', 'fista'}.
    tol : float
        stopping rule tolerance. Default is 1e-5.
    max_iter : int
        maximum number of iterations. Default is 1e4.
    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.
        This parameter is ignored when ``fit_intercept`` is set to ``False``.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling
        ``fit`` on an estimator with ``normalize=False``.
    return_iter : bool
        return the number of iterations before convergence
    """
    def __init__(self, alpha=1.0, groups=None, algorithm='FISTA',
                 fit_intercept=True, tol=1e-5, max_iter=10000, copy_X=True,
                 normalize=False, return_iter=False):
        self.alpha = alpha
        self.groups = groups
        self.algorithm = algorithm
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.return_iter = return_iter
        self.copy_X = copy_X
        self.normalize = normalize

    def fit(self, X, y, check_input=True):
        """Fit model with proximal gradient method.

         Parameters
        -----------
        X : ndarray, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,)
            Target
        check_input : bool
            perform input check  (default = True)
        """
        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator from scikit-learn", stacklevel=2)

        # We expect X and y to be float64 or float32 Fortran ordered arrays
        # when bypassing checks
        if check_input:
            X, y = check_X_y(X, y, accept_sparse=False,
                             order='C', dtype=[np.float64, np.float32],
                             copy=self.copy_X and self.fit_intercept,
                             multi_output=False, y_numeric=True)
            y = check_array(y, order='C', copy=False, dtype=X.dtype.type,
                            ensure_2d=False)

        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, None, False, self.normalize,
                     self.fit_intercept, copy=False)

        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_samples, n_features = X.shape
        # n_targets = y.shape[1]

        # Define group-lasso minimizer
        args = {'loss': 'square',
                'penalty': 'group-lasso',
                'groups': self.groups,
                'tau': self.alpha,
                'tol': self.tol,
                'max_iter': self.max_iter
                }

        if self.algorithm.lower() == 'fista':
            from minimal.optimization import FISTA
            minimizer = partial(FISTA, **args)
        elif self.algorithm.lower() == 'ista':
            from minimal.optimization import ISTA
            minimizer = partial(ISTA, **args)
        else:
            raise NotImplementedError('algorithm must be '
                                      'in {}.'.format(__algorithms__))

        # Run the optimization algorithm
        if self.return_iter:
            self.coef_, _, self.n_iter = minimizer(data=X, labels=y)
        else:
            self.coef_, _ = minimizer(data=X, labels=y)
        self.coef_ = self.coef_.ravel()

        # Set intercept
        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into X.dtype
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        return self


class GroupLassoClassifier(LinearClassifierMixin, LinearModel):
    """Linear classification with Group Lasso penalty as regularizer.

     Minimizes the objective function::

            (1 / n_samples) * L(y, Xw) + alpha * GL(w)

    where ::

            L(y, XW)

    is a suitable loss function (square or logistic) and ::

            GL(w) = ||( ||group_0||^2_2, ... ,||group_G||^2_2])||_2

    is the Group Lasso penalty.

    Parameters
    ----------
    alpha : float
        regularization parameter
    groups : list of lists (used only for group-lasso)
        the outer list represents the groups and the
        inner lists represent the variables in the groups. E.g. ``[[1, 2],
        [2, 3]]`` contains two groups ``([1, 2]`` and ``[2, 3])`` with
        variable 1 and 2 in the first group and variables 2 and 3 in the
        second group.
    loss : string
        the loss function to use, it must be either ``'square'`` (default) or
        ``'logit'``
    algorithm : string
        the selected minimization algorithm in {'ista', 'fista'}.
    tol : float
        stopping rule tolerance. Default is 1e-5.
    max_iter : int
        maximum number of iterations. Default is 1e4.
    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.
        This parameter is ignored when ``fit_intercept`` is set to ``False``.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling
        ``fit`` on an estimator with ``normalize=False``.
    return_iter : bool
        return the number of iterations before convergence
    """
    def __init__(self, alpha=1.0, groups=None, loss='square', algorithm='FISTA',
                 fit_intercept=True, tol=1e-5, max_iter=10000, copy_X=True,
                 normalize=False, return_iter=False):
        self.alpha = alpha
        self.groups = groups
        self.loss = loss
        self.algorithm = algorithm
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.return_iter = return_iter
        self.copy_X = copy_X
        self.normalize = normalize
        self.class_weight = None  # TODO support class weight in future

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit model with proximal gradient method.

         Parameters
        -----------
        X : ndarray, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,)
            Target
        sample_weight : float or numpy array of shape (n_samples,)
            Sample weight (unused, added for consistency only)
        check_input : bool
            perform input check (default = True)
        """
        self._label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        Y = self._label_binarizer.fit_transform(y)
        if not self._label_binarizer.y_type_.startswith('multilabel'):
            y = column_or_1d(y, warn=True)
        else:
            # we don't (yet) support multi-label classification in GL
            raise ValueError(
                "%s doesn't support multi-label classification" % (
                    self.__class__.__name__))

        if self.class_weight:
            if sample_weight is None:
                sample_weight = 1.
            # modify the sample weights with the corresponding class weight
            sample_weight = (sample_weight *
                             compute_sample_weight(self.class_weight, y))

        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator from scikit-learn", stacklevel=2)

        # We expect X and y to be float64 or float32 Fortran ordered arrays
        # when bypassing checks
        if check_input:
            X, y = check_X_y(X, y, accept_sparse=False,
                             order='C', dtype=[np.float64, np.float32],
                             copy=self.copy_X and self.fit_intercept,
                             multi_output=False, y_numeric=True)
            y = check_array(y, order='C', copy=False, dtype=X.dtype.type,
                            ensure_2d=False)

        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, None, False, self.normalize,
                     self.fit_intercept, copy=False)

        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_samples, n_features = X.shape
        # n_targets = y.shape[1]

        # Define group-lasso minimizer
        args = {'loss': self.loss,
                'penalty': 'group-lasso',
                'groups': self.groups,
                'tau': self.alpha,
                'tol': self.tol,
                'max_iter': self.max_iter
                }

        if self.algorithm.lower() == 'fista':
            from minimal.optimization import FISTA
            minimizer = partial(FISTA, **args)
        elif self.algorithm.lower() == 'ista':
            from minimal.optimization import ISTA
            minimizer = partial(ISTA, **args)
        else:
            raise NotImplementedError('algorithm must be '
                                      'in {}.'.format(__algorithms__))

        # Run the optimization algorithm
        if self.return_iter:
            self.coef_, _, self.n_iter = minimizer(data=X, labels=y)
        else:
            self.coef_, _ = minimizer(data=X, labels=y)
        # self.coef_ = self.coef_.ravel()
        self.coef_ = self.coef_.T

        # Set intercept
        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into X.dtype
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        return self

    @property
    def classes_(self):
        return self._label_binarizer.classes_

class NNMRegressor(LinearModel, RegressorMixin):
    """Multi-task linear regression with Nuclear-norm as regularizer.

     Minimizes the objective function::

            (1 / n_samples) * L(Y, XW) + alpha * ||W||_*

    where ::

            L(y, XW)

    is a square-loss function and ::

            ||W||_* = trace(sqrt(W^T W))

    is the Nuclear (trace)-norm penalty.

    Parameters
    ----------
    alpha : float
        regularization parameter
    algorithm : string
        the selected minimization algorithm in {'ista', 'fista'}.
    tol : float
        stopping rule tolerance. Default is 1e-5.
    max_iter : int
        maximum number of iterations. Default is 1e4.
    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.
        This parameter is ignored when ``fit_intercept`` is set to ``False``.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling
        ``fit`` on an estimator with ``normalize=False``.
    return_iter : bool
        return the number of iterations before convergence
    """
    def __init__(self, alpha=1.0, algorithm='FISTA', fit_intercept=True,
                 tol=1e-5, max_iter=10000, copy_X=True, normalize=False,
                 return_iter=False):
        self.alpha = alpha
        self.loss = 'square'
        self.algorithm = algorithm
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.return_iter = return_iter
        self.copy_X = copy_X
        self.normalize = normalize
        self.class_weight = None  # TODO support class weight in future

    def fit(self, X, y, check_input=True):
        """Fit model with proximal gradient method.

         Parameters
        -----------
        X : ndarray, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,)
            Target
        check_input : bool
            perform input check  (default = True)
        """
        if self.alpha == 0:
            warnings.warn("With alpha=0, this algorithm does not converge "
                          "well. You are advised to use the LinearRegression "
                          "estimator from scikit-learn", stacklevel=2)

        # X and y must be of type float64
        X = check_array(X, dtype=np.float64, order='F',
                        copy=self.copy_X and self.fit_intercept)
        y = check_array(y, dtype=np.float64, ensure_2d=False)

        if y.ndim == 1:
            raise ValueError("For mono-task outputs, use Lasso or Elastic-Net")

        n_samples, n_features = X.shape
        _, n_tasks = y.shape

        if n_samples != y.shape[0]:
            raise ValueError("X and y have inconsistent dimensions (%d != %d)"
                             % (n_samples, y.shape[0]))

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, self.fit_intercept, self.normalize, copy=False)

        # Define group-lasso minimizer
        args = {'loss': self.loss,
                'penalty': 'trace',
                'tau': self.alpha,
                'tol': self.tol,
                'max_iter': self.max_iter
                }

        if self.algorithm.lower() == 'fista':
            from minimal.optimization import FISTA
            minimizer = partial(FISTA, **args)
        elif self.algorithm.lower() == 'ista':
            from minimal.optimization import ISTA
            minimizer = partial(ISTA, **args)
        else:
            raise NotImplementedError('algorithm must be '
                                      'in {}.'.format(__algorithms__))

        # Run the optimization algorithm
        if self.return_iter:
            self.coef_, _, self.n_iter = minimizer(data=X, labels=y)
        else:
            self.coef_, _ = minimizer(data=X, labels=y)
        self.coef_ = self.coef_.ravel()

        # Set intercept
        self._set_intercept(X_offset, y_offset, X_scale)

        # workaround since _set_intercept will cast self.coef_ into X.dtype
        self.coef_ = np.asarray(self.coef_, dtype=X.dtype)

        return self
