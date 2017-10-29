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
""" It is an interface for the 'MixedLM' training model (Mixed Linear Model).
"""

from typing import Dict, List, Any, TypeVar
from Stats.Stats import Stats
import statsmodels.api as sm
import sys

PandasDataFrame = TypeVar('DataFrame')
StatsmodelsMixedLM = TypeVar('MixedLM')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class _MixedLinearModel(Stats):
    def __init__(self):
        """Initialise the objects and constants.
        """
        super(self.__class__, self).__init__()
        self._logger.debug(__name__)
        self._logger.debug("Run Mixed Linear Model.")

    def train(self,
              features_indep_df: PandasDataFrame,
              feature_target: List,
              model_labals: List=[0, 1],
              **kwargs: Any) -> StatsmodelsMixedLM:
        """Perform the training, using the Mixed Linear Model.
        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param model_labals: the target labels (default [0, 1]).
        :param kwargs: any other arguments that the selected reader may accept.
        :return: the trained model.
        """
        self._logger.debug("Train " + __name__)
        if 'groups' not in kwargs.keys():
            self._logger.error(__name__ + " - " + " function argument is missing: 'groups'.")
            sys.exit()

        groups = features_indep_df[kwargs['groups']]
        exog = features_indep_df.drop(kwargs['groups'], axis=1)
        exog['Intercept'] = 1

        model_train = sm.MixedLM(endog=feature_target,
                                 exog=exog,
                                 groups=groups,
                                 exog_re=exog['Intercept'])
        model_train = model_train.fit()
        print(model_train.summary())
        return model_train

    def train_summaries(self,
                        model_train: StatsmodelsMixedLM) -> Dict:
        """Produce the training summary.
        :param model_train: the instance of the trained model.
        :return: the training summary.
        """
        self._logger.debug("Summarise " + __name__)
        summaries = dict()
        # todo: summaries
        return summaries

    def plot(self,
             model_train: StatsmodelsMixedLM,
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
