#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from Stats.Stats import Stats
from sklearn import naive_bayes

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class _NaiveBayes(Stats):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._logger.debug(__name__)
        self._logger.info("Running Naive Bayes")

    def train(self, features_indep_df, feature_target, model_labals, **kwargs):
        """
        kwargs:
        alpha=1.0, fit_prior=True, class_prior=None
        """
        self._logger.debug(__name__)

        model_train = naive_bayes.MultinomialNB(**kwargs)
        model_train.fit(features_indep_df.values, feature_target)
        return model_train

    def train_summaries(self, model_train):
        self._logger.debug(__name__)
        summaries = dict()
        summaries['class_log_prior_'] = model_train.class_log_prior_
        summaries['intercept_'] = model_train.intercept_
        summaries['feature_log_prob_'] = model_train.feature_log_prob_
        summaries['coef_'] = model_train.coef_
        summaries['class_count_'] = model_train.class_count_
        summaries['feature_count_'] = model_train.feature_count_
        return summaries
