#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from Stats.Stats import Stats
from sklearn import tree

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class _DecisionTreeClassifier(Stats):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._logger.debug(__name__)
        self._logger.info("Running Decision Tree Classifier")

    def train(self, features_indep_df, feature_target, model_labals, **kwargs):
        """
        kwargs:
        criterion='gini', splitter='best', max_depth=None, min_samples_split=30,
        min_samples_leaf=30, min_weight_fraction_leaf=0.0, max_features=None,
        random_state=None, max_leaf_nodes=None, class_weight=None, presort=False
        """
        self._logger.debug(__name__)

        model_train = tree.DecisionTreeClassifier(**kwargs)
        model_train.fit(features_indep_df.values, feature_target)
        return model_train

    def train_summaries(self, model_train):
        self._logger.debug(__name__)
        summaries = dict()
        summaries['classes_'] = model_train.classes_
        summaries['feature_importances_'] = model_train.feature_importances_
        summaries['max_features_'] = model_train.max_features_
        summaries['n_classes_'] = model_train.n_classes_
        summaries['n_features_'] = model_train.n_features_
        summaries['n_outputs_'] = model_train.n_outputs_
        summaries['tree_summaries'] = model_train.tree_
        return summaries
