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
""" It is the parent class of the developed training models.
It includes abstract methods, the prediction methods and a set of functions for producing statistical summaries for
the prediction models.
"""

from typing import Dict, List, TypeVar, Any
from sklearn import metrics
import sys
import abc
import pandas as pd
import numpy as np
import logging
from collections import OrderedDict
from Configs.CONSTANTS import CONSTANTS

PandasDataFrame = TypeVar('DataFrame')
CollectionsOrderedDict = TypeVar('OrderedDict')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class Stats:
    def __init__(self):
        """Initialise the objects and constants.
        """
        self._logger = logging.getLogger(CONSTANTS.app_name)
        self._logger.debug(__name__)

    @abc.abstractmethod
    def train(self,
              features_indep_df: PandasDataFrame,
              feature_target: List,
              model_labals: List,
              **kwargs: Any) -> Any:
        """Perform the training, using the defined model.
        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param model_labals: the target labels (default [0, 1]).
        :param kwargs: the training model input arguments.
        :return: the trained model.
        """
        pass

    @abc.abstractmethod
    def train_summaries(self,
                        model_train: Any) -> Dict:
        """Produce the training summary.
        :param model_train: the instance of the trained model.
        :return: the training summary.
        """
        pass

    @abc.abstractmethod
    def plot(self,
             model_train: Any,
             feature_names: List,
             class_names: List) -> Any:
        """Plot the tree diagram.
        :param model_train: the instance of the trained model.
        :param feature_names: the names of input features.
        :param class_names: the predicted class labels.
        :return: the model graph.
        """
        pass

    def predict(self,
                model_train: Any,
                features_indep_df: PandasDataFrame) -> Any:
        """Predict probability of labels, using the training model.
        :param model_train: the instance of the trained model.
        :param features_indep_df: the independent features, which are inputted into the model.
        :return: the predicted probabilities, and the predicted labels.
        """
        self._logger.debug("Predict.")
        model_predict = dict()
        features_indep = features_indep_df.values

        model_predict['pred'] = model_train.predict(features_indep)
        model_predict['score'] = model_train.predict_proba(features_indep)[:, 1]
        return model_predict

    def predict_summaries(self,
                          model_predict: PandasDataFrame,
                          feature_target: List) -> CollectionsOrderedDict:
        """Produce summary statistics for the prediction performance.
        :param model_predict: the predicted probabilities, and the predicted labels.
        :param feature_target: the target feature, which is being estimated.
        :return: the prediction summaries.
        """
        self._logger.debug("Produce a prediction summary statistic.")
        summaries = OrderedDict()

        # Accuracy classification score.
        summaries['accuracy_score'] = metrics.accuracy_score(
            y_true=feature_target, y_pred=model_predict['pred'])
        # Compute average precision (AP) from prediction scores
        summaries['average_precision_score'] = metrics.average_precision_score(
            y_true=feature_target, y_score=model_predict['score'])
        # The brier_score_loss function computes the Brier score for binary classes.
        summaries['brier_score_loss'] = metrics.brier_score_loss(
            y_true=feature_target, y_prob=model_predict['score'])
        # Build a text report showing the main classification metrics
        summaries['classification_report'] = metrics.classification_report(
            y_true=feature_target, y_pred=model_predict['pred'])
        # The confusion_matrix function evaluates classification accuracy by computing the confusion matrix.
        summaries['confusion_matrix'] = metrics.confusion_matrix(
            y_true=feature_target, y_pred=model_predict['pred'])
        # Compute the F1 score, also known as balanced F-score or F-measure
        summaries['f1_score'] = metrics.f1_score(
            y_true=feature_target, y_pred=model_predict['pred'], average='binary')
        # Compute the F-beta score
        summaries['fbeta_score'] = metrics.fbeta_score(
            y_true=feature_target, y_pred=model_predict['pred'], average='binary', beta=0.5)
        # Compute the average Hamming loss.
        summaries['hamming_loss'] = metrics.hamming_loss(
            y_true=feature_target, y_pred=model_predict['pred'])
        # Jaccard similarity coefficient score
        summaries['jaccard_similarity_score'] = metrics.jaccard_similarity_score(
            y_true=feature_target, y_pred=model_predict['pred'])
        # Log loss, aka logistic loss or cross-entropy loss.
        summaries['log_loss'] = metrics.log_loss(
            y_true=feature_target, y_pred=model_predict['pred'])
        # Compute the Matthews correlation coefficient (MCC) for binary classes
        summaries['matthews_corrcoef'] = metrics.matthews_corrcoef(
            y_true=feature_target, y_pred=model_predict['pred'])
        # Compute precision, recall, F-measure and support for each class
        summaries['precision_recall_fscore_support'] = metrics.precision_recall_fscore_support(
            y_true=feature_target, y_pred=model_predict['pred'])
        # Compute the precision
        summaries['precision_score'] = metrics.precision_score(
            y_true=feature_target, y_pred=model_predict['pred'])
        # Compute the recall
        summaries['recall_score'] = metrics.recall_score(
            y_true=feature_target, y_pred=model_predict['pred'])
        # Compute Area Under the Curve (AUC) from prediction scores
        summaries['roc_auc_score'] = metrics.roc_auc_score(
            y_true=feature_target, y_score=model_predict['score'])
        # The zero_one_loss function computes the sum or the average of the 0-1 classification loss over n_samples.
        summaries['zero_one_loss'] = metrics.zero_one_loss(
            y_true=feature_target, y_pred=model_predict['pred'])
        return summaries

    def predict_summaries_short(self,
                                predict_score: List,
                                feature_target: List,
                                cutoff: float=0.5) -> Dict:
        """Produce a shortsummary statistics for the prediction performance.
        :param model_predict: the predicted probabilities, and the predicted labels.
        :param feature_target: the target feature, which is being estimated.
        :return: the prediction summaries.
        """
        self._logger.debug("Produce a short prediction summary statistic.")
        summaries = OrderedDict()
        predict_label = np.array([0 if i > 0 and i < cutoff else 1 for i in predict_score])

        # Accuracy classification score.
        summaries['accuracy_score'] = metrics.accuracy_score(
            y_true=feature_target, y_pred=predict_label)
        # Compute average precision (AP) from prediction scores
        summaries['average_precision_score'] = metrics.average_precision_score(
            y_true=feature_target, y_score=predict_score)
        # The brier_score_loss function computes the Brier score for binary classes.
        summaries['brier_score_loss'] = metrics.brier_score_loss(
            y_true=feature_target, y_prob=predict_score)
        # Build a text report showing the main classification metrics
        summaries['classification_report'] = metrics.classification_report(
            y_true=feature_target, y_pred=predict_label)
        # The confusion_matrix function evaluates classification accuracy by computing the confusion matrix.
        summaries['confusion_matrix'] = metrics.confusion_matrix(
            y_true=feature_target, y_pred=predict_label)
        # Compute the F1 score, also known as balanced F-score or F-measure
        summaries['f1_score'] = metrics.f1_score(
            y_true=feature_target, y_pred=predict_label, average='binary')
        # Compute the F-beta score
        summaries['fbeta_score'] = metrics.fbeta_score(
            y_true=feature_target, y_pred=predict_label, average='binary', beta=0.5)
        # Compute the average Hamming loss.
        summaries['hamming_loss'] = metrics.hamming_loss(
            y_true=feature_target, y_pred=predict_label)
        # Jaccard similarity coefficient score
        summaries['jaccard_similarity_score'] = metrics.jaccard_similarity_score(
            y_true=feature_target, y_pred=predict_label)
        # Log loss, aka logistic loss or cross-entropy loss.
        summaries['log_loss'] = metrics.log_loss(
            y_true=feature_target, y_pred=predict_label)
        # Compute the Matthews correlation coefficient (MCC) for binary classes
        summaries['matthews_corrcoef'] = metrics.matthews_corrcoef(
            y_true=feature_target, y_pred=predict_label)
        # Compute precision, recall, F-measure and support for each class
        summaries['precision_recall_fscore_support'] = metrics.precision_recall_fscore_support(
            y_true=feature_target, y_pred=predict_label)
        # Compute the precision
        summaries['precision_score'] = metrics.precision_score(
            y_true=feature_target, y_pred=predict_label)
        # Compute the recall
        summaries['recall_score'] = metrics.recall_score(
            y_true=feature_target, y_pred=predict_label)
        # Compute Area Under the Curve (AUC) from prediction scores
        summaries['roc_auc_score'] = metrics.roc_auc_score(
            y_true=feature_target, y_score=predict_score)
        # The zero_one_loss function computes the sum or the average of the 0-1 classification loss over n_samples.
        summaries['zero_one_loss'] = metrics.zero_one_loss(
            y_true=feature_target, y_pred=predict_label)
        return summaries

    def predict_summaries_cutoffs(self,
                                  predict_score: List,
                                  feature_target: List,
                                  cutoff: float=0.5,
                                  epsilon: float=0.000000000001) -> PandasDataFrame:
        """Produce short summary statistics for a particular cut-off point.
        :param predict_score: the predicted probability.
        :param feature_target: the target feature, which is being estimated.
        :param cutoff: the risk cut-off point.
        :param epsilon: the epsilon value that is used to avoid infinity values.
        :return: the summary statistics for the cut-off point.
        """
        self._logger.debug("Produce a short prediction summary statistic for a cut-off.")
        summaries = pd.DataFrame({'cutoff': [None], 'TP': [None], 'FP': [None], 'TN': [None], 'FN': [None],
                                  'Accuracy': [None], 'Precision': [None], 'Recall': [None],
                                  'Specificity': [None], 'F1-score': [None], 'AUC ROC': [None]})

        if len(predict_score) != len(feature_target):
            self._logger.error(__name__ + " - different array sizes.")
            sys.exit()

        df = pd.DataFrame({"score": predict_score, "target": feature_target})
        df = df.astype(dtype={"score": pd.Series(dtype='f2'), "target": pd.Series(dtype='i1')})
        tp = len(df[(df["score"] >= cutoff) & (df["target"] == 1)])
        fp = len(df[(df["score"] >= cutoff) & (df["target"] == 0)])
        tn = len(df[(df["score"] < cutoff) & (df["target"] == 0)])
        fn = len(df[(df["score"] < cutoff) & (df["target"] == 1)])

        # Cut-Off point
        summaries['cutoff'][0] = cutoff
        # True Positive (TP)
        summaries['TP'][0] = tp
        # False Positive (FP)
        summaries['FP'][0] = fp
        # True Negative (TN)
        summaries['TN'][0] = tn
        # False Negative (FN)
        summaries['FN'][0] = fn
        # Accuracy
        summaries['Accuracy'][0] = (tp + tn) / (len(df["score"]) + epsilon)
        # Precision (PPV)
        summaries['Precision'][0] = tp / ((tp + fp) + epsilon)
        # Recall (Sensitivity)
        summaries['Recall'][0] = tp / ((tp + fn) + epsilon)
        # Specificity
        summaries['Specificity'][0] = tn / ((tn + fp) + epsilon)
        # F1-score
        summaries['F1-score'][0] = (2 * tp) / ((2 * tp + fp + fn) + epsilon)
        # AUC of Receiver operating characteristic (ROC).
        summaries['AUC ROC'][0] = metrics.roc_auc_score(y_true=df["target"], y_score=df["score"])

        summaries = summaries.reset_index(drop=True)
        return summaries.reindex_axis(['cutoff', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall',
                                       'Specificity', 'F1-score', 'AUC ROC'], axis=1)

    def predict_summaries_cutoffs_table(self,
                                        predict_score: List,
                                        feature_target: List,
                                        cutoffs: List=np.arange(0, 1.05, 0.05)) -> PandasDataFrame:
        """Produce a summary statistics table for a range of cut-off points.
        :param predict_score: the predicted probability.
        :param feature_target: the target feature, which is being estimated.
        :param cutoffs: a list of risk cut-off points.
        :return: the summary statistics table for the cut-off points.
        """
        self._logger.debug("Produce  a summary statistics table for a range of cut-off points.")
        summaries = pd.DataFrame({'cutoff': [] * len(cutoffs), 'TP': [] * len(cutoffs), 'FP': [] * len(cutoffs),
                                  'TN': [] * len(cutoffs), 'FN': [] * len(cutoffs),
                                  'Accuracy': [] * len(cutoffs), 'Precision': [] * len(cutoffs),
                                  'Recall': [] * len(cutoffs), 'Specificity': [] * len(cutoffs),
                                  'F1-score': [] * len(cutoffs), 'AUC ROC': [] * len(cutoffs)})
        for cutoff in cutoffs:
            summaries = summaries.append(self.predict_summaries_cutoffs(predict_score, feature_target, cutoff))
        summaries = summaries.reset_index(drop=True)
        return summaries.reindex_axis(['cutoff', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall',
                                       'Specificity', 'F1-score', 'AUC ROC'], axis=1)
