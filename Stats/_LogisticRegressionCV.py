#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from Stats.Stats import Stats
from sklearn import linear_model

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class _LogisticRegressionCV(Stats):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._logger.debug(__name__)
        self._logger.info("Running Logistic Regression Cross-Validation")

    def train(self, features_indep, feature_target, model_labals, **kwargs):
        """
        kwargs:
        Cs=10, fit_intercept=True, cv=None, dual=False, penalty='l2', scoring=None, solver='lbfgs',
        tol=0.0001, max_iter=100, class_weight=None, n_jobs=-1, verbose=0, refit=True, intercept_scaling=1.0,
        multi_class='ovr', random_state=None
        """
        self._logger.debug(__name__)

        model_train = linear_model.LogisticRegressionCV(**kwargs)
        model_train.fit(features_indep, feature_target)
        return model_train

    def train_summaries(self, model_train):
        self._logger.debug(__name__)
        summaries = dict()

        # Coefficient of the features in the decision function.
        summaries["coef_"] = model_train.coef_
        # Intercept (a.k.a. bias) added to the decision function.
        summaries["intercept_"] = model_train.intercept_
        # Actual number of iterations for all classes. If binary or multinomial, it returns only 1 element.
        summaries["n_iter_"] = model_train.n_iter_
        return summaries
