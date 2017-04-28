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


class _RandomForestClassifier(Stats):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._logger.debug(__name__)
        self._logger.info("Running Random Forest Classifier")

    def train(self, features_indep_df, feature_target, model_labals, **kwargs):
        """
        kwargs:
        n_estimators=20, criterion='gini', max_depth=None, min_samples_split=100,
        min_samples_leaf=50, min_weight_fraction_leaf=0.0, max_features='auto',
        max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None,
        verbose=0, warm_start=False, class_weight="balanced_subsample"
        """
        self._logger.debug(__name__)

        model_train = ensemble.RandomForestClassifier(**kwargs)
        model_train.fit(features_indep_df.values, feature_target)
        return model_train

    def train_summaries(self, model_train):
        self._logger.debug(__name__)
        summaries = dict()
        summaries['estimators_'] = model_train.estimators_
        summaries['classes_'] = model_train.classes_
        summaries['n_classes_'] = model_train.n_classes_
        summaries['n_features_'] = model_train.n_features_
        summaries['n_outputs_'] = model_train.n_outputs_
        summaries['feature_importances_'] = model_train.feature_importances_
        return summaries
