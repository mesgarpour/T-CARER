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
"""It applies the training functions
"""

from typing import Dict, List, TypeVar, Any
from Stats._LogisticRegression import _LogisticRegression
from Stats._LogisticRegressionCV import _LogisticRegressionCV
from Stats._MixedLinearModel import _MixedLinearModel
from Stats._RandomForestClassifier import _RandomForestClassifier
from Stats._GradientBoostingClassifier import _GradientBoostingClassifier
from Stats._DecisionTreeClassifier import _DecisionTreeClassifier
from Stats._KNeighborsClassifier import _KNeighborsClassifier
from Stats._NaiveBayes import _NaiveBayes
from Stats._NeuralNetwork import _NeuralNetwork
from ReadersWriters.ReadersWriters import ReadersWriters
from Configs.CONSTANTS import CONSTANTS
from sklearn.model_selection import cross_val_score
import numpy as np
import sys
import logging

PandasDataFrame = TypeVar('DataFrame')
NumpyNDArray = TypeVar('ndarray')
CollectionsOrderedDict = TypeVar('OrderedDict')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class TrainingMethod:
    def __init__(self,
                 method_name: str,
                 path: str=None,
                 title: str=None):
        """Initialise the objects and constants.
        :param method_name: the training method that will be used
        (options: {'lr': Logistic Regression, 'lr_cv': Logistic Regression with Cross-Validation,
        'mlm': Mixed Linear Model, 'rfc': Random Forest Classifier, 'gbc': Gradient Boosting Classifier,
        'dtc' Decision Tree Classifier, 'knc': K-Nearest Neighbors Classifier, 'nb': Multinomial Naive Bayes,
        'nn': Multi-Layer Perceptron (MLP) Neural Network}).
        :param path: the directory path of the saved trained model file, using this application (if applicable).
        :param title: the file name of the saved trained model file, using this application
        """
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)

        self.__readers_writers = ReadersWriters()
        self.__method = None
        self.method_name = method_name
        self.model_labels = None
        self.model_train = None
        self.model_predict = dict()
        self.model_cross_validate = None
        if method_name is not None:
            self.__init__method(method_name)
        else:
            self.load(path, title)

    def __init__method(self,
                       method_name: str,
                       model_labels: List=None,
                       model_train: Any=None,
                       model_predict: Dict=None,
                       model_cross_validate: NumpyNDArray=None):
        """Initialise the selected training method.
        :param method_name: the training method that will be used
        (options: {'lr': Logistic Regression, 'lr_cv': Logistic Regression with Cross-Validation,
        'mlm': Mixed Linear Model, 'rfc': Random Forest Classifier, 'gbc': Gradient Boosting Classifier,
        'dtc' Decision Tree Classifier, 'knc': K-Nearest Neighbors Classifier, 'nb': Multinomial Naive Bayes,
        'nn': Multi-Layer Perceptron (MLP) Neural Network}).
        :param model_labels: the features names to be inputted into the model.
        Note: the order of features will be preserved internally.
        :param model_train: the training model.
        :param model_predict: the prediction outputs.
        :param model_cross_validate: the cross-validation model.
        """
        self.__logger.debug("Initialise the training method.")
        if method_name == "lr":
            self.__method = _LogisticRegression()
        elif method_name == "lr_cv":
            self.__method = _LogisticRegressionCV()
        elif method_name == "mlm":
            self.__method = _MixedLinearModel()
        elif method_name == "rfc":
            self.__method = _RandomForestClassifier()
        elif method_name == "gbc":
            self.__method = _GradientBoostingClassifier()
        elif method_name == "dtc":
            self.__method = _DecisionTreeClassifier()
        elif method_name == "knc":
            self.__method = _KNeighborsClassifier()
        elif method_name == "nb":
            self.__method = _NaiveBayes()
        elif method_name == "nn":
            self.__method = _NeuralNetwork()
        else:
            self.__logger.error(__name__ + " - Invalid training method: " + str(method_name))
            sys.exit()

        self.model_labels = model_labels
        self.model_train = model_train
        self.model_predict = dict() if model_predict is None else model_predict
        self.model_cross_validate = model_cross_validate

    def train(self,
              features_indep_df: PandasDataFrame,
              feature_target: List,
              **kwargs: Any) -> Any:
        """Perform the training, using the selected method.
        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param kwargs: the training method's argument.
        :return: the trained model.
        """
        self.__logger.debug("Train.")
        self.model_labels = list(features_indep_df.columns.values)
        self.model_train = self.__method.train(
            features_indep_df[self.model_labels], feature_target, self.model_labels, **kwargs)
        return self.model_train

    def plot(self) -> Any:
        """Plot the tree diagram.
        :return: the model graph.
        """
        self.__logger.debug("Plot.")
        return self.__method.plot(self.model_train, self.model_labels, ["True", "False"])

    def train_summaries(self) -> Any:
        """ Produce the training summary.
        :return: the training summary.
        """
        self.__logger.debug("Summarise training model.")
        return self.__method.train_summaries(self.model_train)

    def predict(self,
                features_indep_df: PandasDataFrame,
                sample_name: str) -> PandasDataFrame:
        """Predict probability of labels, using the training model.
        :param features_indep_df: the independent features, which are inputted into the model.
        :param sample_name: the sample to predict(e.g. 'train', 'test', 'validate').
        :return: the predicted probabilities, and the predicted labels.
        """
        self.__logger.debug("Predict.")
        self.model_predict[sample_name] = self.__method.predict(self.model_train, features_indep_df[self.model_labels])
        return self.model_predict[sample_name]

    def predict_summaries(self,
                          feature_target: List,
                          sample_name: str) -> CollectionsOrderedDict:
        """roduce summary statistics for the prediction performance.
        :param feature_target: the target feature, which is being estimated.
        :param sample_name: the sample to predict(e.g. 'train', 'test', 'validate').
        :return: the prediction summaries.
        """
        self.__logger.debug("Summarise predictions.")
        self.model_predict[sample_name]['target'] = feature_target
        return self.__method.predict_summaries(self.model_predict[sample_name], feature_target)

    def predict_summaries_risk_bands(self,
                                     feature_target: List,
                                     sample_name: str,
                                     cutoffs: List=np.arange(0, 1.05, 0.05)) -> CollectionsOrderedDict:
        """Produce a summary statistics table for a range of cut-off points.
        :param feature_target: the target feature, which is being estimated.
        :param sample_name: the sample to predict(e.g. 'train', 'test', 'validate').
        :param cutoffs: a list of risk cut-off points.
        :return: the summary statistics table for the cut-off points.
        """
        self.__logger.debug("Summarise predictions.")
        self.model_predict[sample_name]['target'] = feature_target
        return self.__method.predict_summaries_cutoffs_table(
            self.model_predict[sample_name]['score'], feature_target, cutoffs)

    def cross_validate(self,
                       features_indep_df: PandasDataFrame,
                       feature_target: List,
                       scoring: str="neg_mean_squared_error",
                       cv: int=10) -> Any:
        """Evaluate the model by performing cross-validation.
        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param scoring: the scoring method (default: 'neg_mean_squared_error').
        :param cv: the cross-validation splitting strategy (optional).
        :return: the cross-validation summary
        """
        self.__logger.info("Cross-Validate")

        self.model_cross_validate = cross_val_score(
            self.model_train, features_indep_df[self.model_labels], feature_target, scoring=scoring, cv=cv)
        return self.model_cross_validate

    def cross_validate_summaries(self) -> Any:
        """Produce a summary of the applied cross-validation
        :return: the cross-validation summary
        """
        return self.model_cross_validate

    def save_model(self,
                   path: str,
                   title: str):
        """Save (pickle) the training model, as well as predictions and cross-validations.
        Note: summaries statistics won't not saved.
        :param path: the directory path of the saved trained model file, using this application (if applicable).
        :param title: the file name of the saved trained model file, using this application.
        """
        self.__logger.info("Saving model")
        objects = dict()
        objects['method_name'] = self.method_name
        objects['model_labels'] = self.model_labels
        objects['model_train'] = self.model_train
        objects['model_predict'] = self.model_predict
        objects['model_cross_validate'] = self.model_cross_validate
        self.__readers_writers.save_serialised(path, title, objects=objects)

    def save_model_compressed(self,
                              path: str,
                              title: str):
        """Save (pickle) & compressthe training model, as well as predictions and cross-validations.
        Note: summaries statistics won't not saved.
        :param path: the directory path of the saved trained model file, using this application (if applicable).
        :param title: the file name of the saved trained model file, using this application.
        """
        self.__logger.debug("Save model.")
        objects = dict()
        objects['method_name'] = self.method_name
        objects['model_labels'] = self.model_labels
        objects['model_train'] = self.model_train
        objects['model_predict'] = self.model_predict
        objects['model_cross_validate'] = self.model_cross_validate
        self.__readers_writers.save_serialised_compressed(path, title, objects=objects)

    def load(self,
             path: str,
             title: str):
        """Load (unpickle) the model, which was saved using this application.
        :param path: the directory path of the saved trained model file, using this application (if applicable).
        :param title: the file name of the saved trained model file, using this application
        """
        self.__logger.debug("Load model.")
        objects = self.__readers_writers.load_serialised(path, title)
        try:
            self.__init__method(method_name=objects['method_name'],
                                model_labels=objects['model_labels'],
                                model_train=objects['model_train'],
                                model_predict=objects['model_predict'],
                                model_cross_validate=objects['model_cross_validate'])
        except():
            self.__logger.error(__name__ + " - Invalid field(s) in the model file: " + path)
            sys.exit()
