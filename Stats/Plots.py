#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sklearn import metrics
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import itertools
import numpy as np
import logging
from Configs.CONSTANTS import CONSTANTS

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class Plots:
    def __init__(self):
        self._logger = logging.getLogger(CONSTANTS.app_name)
        self._logger.debug(__name__)

    def confusion_matrix(self, model_predict, feature_target, classes=list([0, 1]),
                         normalize=False, title='Confusion Matrix', cmap="Blues"):
        self._logger.debug(__name__)
        summaries = dict()
        tick_marks = np.arange(len(classes))

        # Compute confusion matrix
        summaries["cnf_matrix"] = confusion_matrix(feature_target, model_predict['score'])
        np.set_printoptions(precision=2)

        # plot metadata
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.clf()
        ax.title(title + ' Average Precision={0:0.2f}'.format(summaries["avg_precision"]))
        ax.ylabel('True label')
        ax.xlabel('Predicted label')
        ax.xticks(tick_marks, classes, rotation=45)
        ax.yticks(tick_marks, classes)
        ax.grid()
        ax.colorbar()

        # plot matrix
        plt.imshow(summaries["cnf_matrix"], interpolation='nearest', cmap=cmap)

        if normalize:
            summaries["cnf_matrix"] = \
                summaries["cnf_matrix"].astype('float') / summaries["cnf_matrix"].sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(summaries["cnf_matrix"])

        thresh = summaries["cnf_matrix"].max() / 2.
        for i, j in itertools.product(range(summaries["cnf_matrix"].shape[0]), range(summaries["cnf_matrix"].shape[1])):
            plt.text(j, i, summaries["cnf_matrix"][i, j],
                     horizontalalignment="center",
                     color="white" if summaries["cnf_matrix"][i, j] > thresh else "black")

        plt.tight_layout()
        return fig, summaries

    def precision_recall(self, model_predict, feature_target, title="Precision-Recall Curve", lw=2):
        self._logger.debug(__name__)
        summaries = dict()

        # calculate
        summaries["precision"], summaries["recall"], _ = metrics.precision_recall_curve(
            feature_target, model_predict['score'])

        # summaries
        summaries["avg_precision"] = metrics.average_precision_score(feature_target, model_predict['score'])

        # plot metadata
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.clf()
        plt.title(title + ' Average Precision={0:0.2f}'.format(summaries["avg_precision"]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid()

        plt.plot(summaries["precision"], summaries["recall"], lw=lw, color='navy', label='Precision-Recall curve')

        plt.legend(loc="lower left")
        return fig, summaries

    def roc(self, model_predict, feature_target, title="ROC Curve", lw=2):
        self._logger.debug(__name__)
        summaries = dict()

        # calculate
        summaries["fpr"], summaries["tpr"], _ = metrics.roc_curve(feature_target, model_predict['score'])

        # summaries
        summaries["roc_auc"] = metrics.auc(summaries["fpr"], summaries["tpr"])

        # plot metadata
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.clf()
        plt.title(title + ' AUC={0:0.2f}'.format(summaries["roc_auc"]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid()

        plt.plot(summaries["fpr"], summaries["tpr"], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % summaries["roc_auc"])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

        plt.legend(loc="lower right")
        return fig, summaries

    def learning_curve(self, estimator, features_indep_df, feature_target,
                       title="Learning Curve", ylim=None, cv=None,
                       n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
        self._logger.debug(__name__)
        summaries = dict()

        # calculate
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, features_indep_df, feature_target,
            cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

        # summaries
        summaries["train_scores_mean"] = np.mean(train_scores, axis=1)
        summaries["train_scores_std"] = np.std(train_scores, axis=1)
        summaries["test_scores_mean"] = np.mean(test_scores, axis=1)
        summaries["test_scores_std"] = np.std(test_scores, axis=1)

        # plot metadata
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.clf()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()

        # plot curves
        plt.fill_between(train_sizes, summaries["train_scores_mean"] - summaries["train_scores_std"],
                         summaries["train_scores_mean"] + summaries["train_scores_std"], alpha=0.1, color="r")
        plt.fill_between(train_sizes, summaries["test_scores_mean"] - summaries["test_scores_std"],
                         summaries["test_scores_mean"] + summaries["test_scores_std"], alpha=0.1, color="g")
        plt.plot(train_sizes, summaries["train_scores_mean"], 'o-', color="r", label="Training score")
        plt.plot(train_sizes, summaries["test_scores_mean"], 'o-', color="g", label="Cross-validation score")

        plt.legend(loc="best")
        return fig, summaries

    def validation_curve(self, estimator, features_indep_df, feature_target, param_name, param_range,
                         title="Learning Curve", ylim=None, cv=None, lw=2, n_jobs=-1):
        self._logger.debug(__name__)
        summaries = dict()

        # train & test
        train_scores, test_scores = validation_curve(
            estimator, features_indep_df, feature_target,
            param_name=param_name, param_range=param_range,
            cv=cv, scoring="accuracy", n_jobs=n_jobs)

        # summaries
        summaries["train_scores_mean"] = np.mean(train_scores, axis=1)
        summaries["train_scores_std"] = np.std(train_scores, axis=1)
        summaries["test_scores_mean"] = np.mean(test_scores, axis=1)
        summaries["test_scores_std"] = np.std(test_scores, axis=1)

        # plot metadata
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.clf()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("$\gamma$")
        plt.ylabel("Score")
        plt.grid()

        # plot curves
        plt.semilogx(param_range, summaries["train_scores_mean"], label="Training score", color="darkorange", lw=lw)
        plt.fill_between(param_range, summaries["train_scores_mean"] - summaries["train_scores_std"],
                         summaries["train_scores_mean"] + summaries["train_scores_std"], alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(param_range, summaries["test_scores_mean"], label="Cross-validation score", color="navy", lw=lw)
        plt.fill_between(param_range, summaries["test_scores_mean"] - summaries["test_scores_std"],
                         summaries["test_scores_mean"] + summaries["test_scores_std"], alpha=0.2, color="navy", lw=lw)

        plt.legend(loc="best")
        return fig, summaries

    def distribution_bar(self, feature, feature_name, title, ylim=[0.0, 1.05]):
        self._logger.debug(__name__)
        uniques = np.unique(feature)
        uniques.sort()

        # plot metadata
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.clf()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(feature_name)
        plt.ylabel("Probability")
        plt.grid()

        # plot curves
        plt.hist(feature, bins=uniques, normed=1, facecolor='green', alpha=0.5)

        return fig

    def distribution_hist(self, feature, feature_name, title, num_bins=50, ylim=[0.0, 1.05]):
        self._logger.debug(__name__)

        # plot metadata
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.clf()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(feature_name)
        plt.ylabel("Probability")
        plt.grid()

        # plot curves# the histogram of the data
        plt.hist(feature, num_bins, normed=1, facecolor='green', alpha=0.5)

        return fig

    def distribution_kde(self, feature, feature_name, title,
                         x_values=None, kernel="gaussian", bandwidth=0.5, ylim=[0.0, 1.05]):
        self._logger.debug(__name__)
        if x_values is None:
            x_values = np.linspace(min(feature), max(feature), len(feature))[:, np.newaxis]
        else:
            x_values = x_values[:, np.newaxis]

        # plot metadata
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.clf()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(feature_name)
        plt.ylabel("Probability")
        plt.grid()

        # plot curves# the histogram of the data
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(np.array(feature)[:, np.newaxis])
        log_dens = kde.score_samples(x_values)
        plt.plot(x_values[:, 0], np.exp(log_dens), '-', label="kernel = '{0}'".format(kernel))

        return fig
