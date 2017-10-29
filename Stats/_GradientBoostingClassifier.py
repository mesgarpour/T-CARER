#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2017 University of Westminster. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" It is an interface for the 'GradientBoostingClassifier' training model (Gradient Boosting Classifier).
"""

from typing import Dict, List, Any, TypeVar
from Stats.Stats import Stats
from sklearn import ensemble

PandasDataFrame = TypeVar('DataFrame')
SklearnGradientBoostingClassifier = TypeVar('GradientBoostingClassifier')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class _GradientBoostingClassifier(Stats):
    def __init__(self):
        """Initialise the objects and constants.
        """
        super(self.__class__, self).__init__()
        self._logger.debug(__name__)
        self._logger.debug("Run Gradient Boosting Classifier.")

    def train(self,
              features_indep_df: PandasDataFrame,
              feature_target: List,
              model_labals: List=[0, 1],
              **kwargs: Any) -> SklearnGradientBoostingClassifier:
        """Perform the training, using the Gradient Boosting Classifier.
        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param model_labals: the target labels (default [0, 1]).
        :param kwargs: loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=30,
        min_samples_leaf=30, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None,
        max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto'
        :return: the trained model.
        """
        self._logger.debug("Train " + __name__)
        model_train = ensemble.GradientBoostingClassifier(**kwargs)
        model_train.fit(features_indep_df.values, feature_target)
        return model_train

    def train_summaries(self,
                        model_train: SklearnGradientBoostingClassifier) -> Dict:
        """Produce the training summary.
        :param model_train: the instance of the trained model.
        :return: the training summary.
        """
        self._logger.debug("Summarise " + __name__)
        summaries = dict()
        summaries['feature_importances_'] = model_train.feature_importances_
        summaries['train_score_'] = model_train.train_score_
        summaries['loss_'] = model_train.loss_
        summaries['init'] = model_train.init
        summaries['estimators_'] = model_train.estimators_
        return summaries

    def plot(self,
             model_train: SklearnGradientBoostingClassifier,
             feature_names: List,
             class_names: List=["True", "False"]):
        """Plot the tree diagram.
        :param model_train: the instance of the trained model.
        :param feature_names: the names of input features.
        :param class_names: the predicted class labels.
        :return: the model graph.
        """
        self._logger.debug("Plot " + __name__)
        # todo: plot
        pass
