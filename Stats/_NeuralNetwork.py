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
""" It is an interface for the 'MLPClassifier' training model (Multi-Layer Perceptron (MLP) Neural Network).
"""

from typing import Dict, List, Any, TypeVar
from Stats.Stats import Stats
from sklearn.neural_network import MLPClassifier

PandasDataFrame = TypeVar('DataFrame')
SklearnMLPClassifier = TypeVar('MLPClassifier')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class _NeuralNetwork(Stats):
    def __init__(self):
        """Initialise the objects and constants.
        """
        super(self.__class__, self).__init__()
        self._logger.debug(__name__)
        self._logger.debug("Running Neural Network.")

    def train(self,
              features_indep_df: PandasDataFrame,
              feature_target: List,
              model_labals: List=[0, 1],
              **kwargs: Any) -> SklearnMLPClassifier:
        """Perform the training, using the Multi-Layer Perceptron (MLP) Neural Network.
        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param model_labals: the target labels (default [0, 1]).
        :param kwargs: solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
        :return: the trained model.
        """
        self._logger.debug("Train " + __name__)
        model_train = MLPClassifier(**kwargs)
        model_train.fit(features_indep_df.values, feature_target)
        return model_train

    def train_summaries(self,
                        model_train: SklearnMLPClassifier) -> Dict:
        """Produce the training summary.
        :param model_train: the instance of the trained model.
        :return: the training summary.
        """
        self._logger.debug("Summarise " + __name__)
        summaries = dict()
        summaries['classes_'] = model_train.classes_
        summaries['loss_'] = model_train.loss_
        summaries['coefs_'] = model_train.coefs_
        summaries['intercepts_'] = model_train.intercepts_
        summaries['n_iter_'] = model_train.n_iter_
        summaries['n_layers_'] = model_train.n_layers_
        summaries['n_outputs_'] = model_train.n_outputs_
        summaries['out_activation_'] = model_train.out_activation_
        return summaries

    def plot(self,
             model_train: SklearnMLPClassifier,
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
