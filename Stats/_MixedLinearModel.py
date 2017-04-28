#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from Stats.Stats import Stats
import statsmodels.api as sm
import sys

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class _MixedLinearModel(Stats):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._logger.debug(__name__)
        self._logger.info("Running Mixed Linear Model")

    def train(self, features_indep_df, feature_target, model_labals, **kwargs):
        """
        kwargs:
        groups=groups
        """
        self._logger.debug(__name__)

        if 'groups' not in kwargs.keys():
            self._logger.error(__name__ + " - " + " function argument is missing: 'groups'")
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

    def train_summaries(self, model_train):
        self._logger.debug(__name__)
        summaries = dict()
        # todo: generate summaries
        return summaries
