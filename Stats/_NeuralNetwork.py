#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from Stats.Stats import Stats
from sklearn.neural_network import MLPClassifier

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class _NeuralNetwork(Stats):
    def __init__(self):
        super(self.__class__, self).__init__()
        self._logger.debug(__name__)
        self._logger.info("Running Neural Network")

    def train(self, features_indep_df, feature_target, model_labals, **kwargs):
        """
        kwargs:
        solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
        """
        self._logger.debug(__name__)
        model_train = MLPClassifier(**kwargs)
        model_train.fit(features_indep_df.values, feature_target)
        return model_train

    def train_summaries(self, model_train):
        self._logger.debug(__name__)
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
