#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sklearn import metrics
import abc
import pandas as pd
import numpy as np
import collections
import logging
from Configs.CONSTANTS import CONSTANTS
import sys

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class Stats:
    def __init__(self):
        self._logger = logging.getLogger(CONSTANTS.app_name)
        self._logger.debug(__name__)

    @abc.abstractmethod
    def train(self, features_indep_df, feature_target, model_labals, **kwargs):
        pass

    @abc.abstractmethod
    def train_summaries(self, model_train):
        pass

    def predict(self, model_train, features_indep_df):
        self._logger.debug(__name__)
        self._logger.info("Predicting")

        model_predict = dict()
        features_indep = features_indep_df.values

        model_predict['pred'] = model_train.predict(features_indep)
        model_predict['score'] = model_train.predict_proba(features_indep)[:, 1]
        model_predict['score_0'] = model_train.predict_proba(features_indep)[:, 0]
        return model_predict

    def predict_summaries(self, model_predict, feature_target):
        self._logger.debug(__name__)
        summaries = collections.OrderedDict()

        # ‘accuracy’
        summaries['accuracy_score'] = metrics.accuracy_score(
            y_true=feature_target, y_pred=model_predict['pred'])
        # ‘average_precision’
        summaries['average_precision_score'] = metrics.average_precision_score(
            y_true=feature_target, y_score=model_predict['score'])
        # 'brier_score_loss'
        summaries['brier_score_loss'] = metrics.brier_score_loss(
            y_true=feature_target, y_prob=model_predict['score'])
        # 'classification report'
        summaries['classification_report'] = metrics.classification_report(
            y_true=feature_target, y_pred=model_predict['pred'])
        # 'confusion_matrix'
        summaries['confusion_matrix'] = metrics.confusion_matrix(
            y_true=feature_target, y_pred=model_predict['pred'])
        # ‘f1’ for binary feature_targets
        summaries['f1_score'] = metrics.f1_score(
            y_true=feature_target, y_pred=model_predict['pred'], average='binary')
        # 'fbeta_score'
        summaries['fbeta_score'] = metrics.fbeta_score(
            y_true=feature_target, y_pred=model_predict['pred'], average='binary', beta=0.5)
        # 'hamming_loss
        summaries['hamming_loss'] = metrics.hamming_loss(
            y_true=feature_target, y_pred=model_predict['pred'])
        # 'jaccard_similarity_score'
        summaries['jaccard_similarity_score'] = metrics.jaccard_similarity_score(
            y_true=feature_target, y_pred=model_predict['pred'])
        # 'log_loss'
        summaries['log_loss'] = metrics.log_loss(
            y_true=feature_target, y_pred=model_predict['pred'])
        # 'matthews_corrcoef'
        summaries['matthews_corrcoef'] = metrics.matthews_corrcoef(
            y_true=feature_target, y_pred=model_predict['pred'])
        # 'precision_recall_fscore_support'
        summaries['precision_recall_fscore_support'] = metrics.precision_recall_fscore_support(
            y_true=feature_target, y_pred=model_predict['pred'])
        # 'precision_score'
        summaries['precision_score'] = metrics.precision_score(
            y_true=feature_target, y_pred=model_predict['pred'])
        # 'recall_score'
        summaries['recall_score'] = metrics.recall_score(
            y_true=feature_target, y_pred=model_predict['pred'])
        # 'roc_auc_score'
        summaries['roc_auc_score'] = metrics.roc_auc_score(
            y_true=feature_target, y_score=model_predict['score'])
        # 'zero_one_loss'
        summaries['zero_one_loss'] = metrics.zero_one_loss(
            y_true=feature_target, y_pred=model_predict['pred'])
        return summaries

    def predict_summaries_cutoffs(self, predict_score, feature_target, cutoff=0.5, epsilon=0.000000000001):
        self._logger.debug(__name__)
        summaries = pd.DataFrame({'cutoff': [None], 'TP': [None], 'FP': [None], 'TN': [None], 'FN': [None],
                                  'Accuracy': [None], 'Precision': [None], 'Recall': [None],
                                  'Specificity': [None], 'F1-score': [None], 'AUC ROC': [None]})

        if len(predict_score) != len(feature_target):
            self._logger.error(__name__ + " - different array sizes")
            sys.exit()

        df = pd.DataFrame({"score": predict_score.copy(True),
                           "target": feature_target.copy(True)})
        df = df.astype({"score": 'f4', "target": 'i4'})
        tp = len(df[(df["score"] >= cutoff) & (df["target"] == 1)])
        fp = len(df[(df["score"] >= cutoff) & (df["target"] == 0)])
        tn = len(df[(df["score"] < cutoff) & (df["target"] == 0)])
        fn = len(df[(df["score"] < cutoff) & (df["target"] == 1)])

        summaries['cutoff'][0] = cutoff
        # ‘tp’
        summaries['TP'][0] = tp
        # ‘fp’
        summaries['FP'][0] = fp
        # ‘tn’
        summaries['TN'][0] = tn
        # ‘fn’
        summaries['FN'][0] = fn
        # ‘accuracy’
        summaries['Accuracy'][0] = (tp + tn) / (len(df["score"]) + epsilon)
        # ‘precision’
        summaries['Precision'][0] = tp / ((tp + fp) + epsilon)
        # ‘recall’
        summaries['Recall'][0] = tp / ((tp + fn) + epsilon)
        # ‘specificity’
        summaries['Specificity'][0] = tn / ((tn + fp) + epsilon)
        # ‘f1’
        summaries['F1-score'][0] = (2 * tp) / ((2 * tp + fp + fn) + epsilon)
        # 'roc_auc'
        summaries['AUC ROC'][0] = metrics.roc_auc_score(y_true=df["target"], y_score=df["score"])
        summaries = summaries.reset_index(drop=True)
        return summaries.reindex_axis(['cutoff', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall',
                                       'Specificity', 'F1-score', 'AUC ROC'], axis=1)

    def predict_summaries_cutoffs_table(self, predict_score, feature_target, cutoffs=np.arange(0, 1.05, 0.05)):
        summaries = pd.DataFrame({'cutoff': [] * len(cutoffs),  'TP': [] * len(cutoffs), 'FP': [] * len(cutoffs),
                                  'TN': [] * len(cutoffs), 'FN': [] * len(cutoffs),
                                  'Accuracy': [] * len(cutoffs), 'Precision': [] * len(cutoffs),
                                  'Recall': [] * len(cutoffs), 'Specificity': [] * len(cutoffs),
                                  'F1-score': [] * len(cutoffs), 'AUC ROC': [] * len(cutoffs)})
        for cutoff in cutoffs:
            summaries = summaries.append(self.predict_summaries_cutoffs(predict_score, feature_target, cutoff))
        summaries = summaries.reset_index(drop=True)
        return summaries.reindex_axis(['cutoff', 'TP', 'FP', 'TN', 'FN', 'Accuracy', 'Precision', 'Recall',
                                       'Specificity', 'F1-score', 'AUC ROC'], axis=1)
