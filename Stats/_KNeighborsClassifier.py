#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from Stats.Stats import Stats
from sklearn import neighbors

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class _KNeighborsClassifier(Stats):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._logger.debug(__name__)
        self._logger.info("Running K-Neighbors Classifier")

    def train(self, features_indep_df, feature_target, model_labals, **kwargs):
        """
        kwargs:
        n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30,
        p=2, metric='minkowski', metric_params=None, n_jobs=-1
        """
        self._logger.debug(__name__)

        model_train = neighbors.KNeighborsClassifier(**kwargs)
        model_train.fit(features_indep_df.values, feature_target)
        return model_train

    def train_summaries(self, model_train):
        self._logger.debug(__name__)
        summaries = dict()
        return summaries
