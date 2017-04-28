#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from Stats.Stats import Stats
from sklearn import ensemble

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class _GradientBoostingClassifier(Stats):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._logger.debug(__name__)
        self._logger.info("Running Gradient Boosting Classifier")

    def train(self, features_indep_df, feature_target, model_labals, **kwargs):
        """
        kwargs:
        loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=30,
        min_samples_leaf=30, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None,
        max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto'
        """
        self._logger.debug(__name__)

        model_train = ensemble.GradientBoostingClassifier(**kwargs)
        model_train.fit(features_indep_df.values, feature_target)
        return model_train

    def train_summaries(self, model_train):
        self._logger.debug(__name__)
        summaries = dict()
        summaries['feature_importances_'] = model_train.feature_importances_
        summaries['train_score_'] = model_train.train_score_
        summaries['loss_'] = model_train.loss_
        summaries['init'] = model_train.init
        summaries['estimators_'] = model_train.estimators_
        return summaries
