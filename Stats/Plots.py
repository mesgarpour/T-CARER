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
"""It consists of a set of custom plots, using Matplotlib and Scikit libraries.
"""

from typing import Dict, List, TypeVar, Any
from sklearn import metrics
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np
import itertools

PandasDataFrame = TypeVar('DataFrame')
MatplotlibFigure = TypeVar('Figure')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class Plots:
    @staticmethod
    def confusion_matrix(predicted_scores: List,
                         feature_target: List,
                         model_labals: List=list([0, 1]),
                         normalize: bool=False,
                         title: str='Confusion Matrix',
                         cmap: str="Blues") -> [MatplotlibFigure, Dict]:
        """Plot the confusion matrix.
        :param predicted_scores: the predicted Scores.
        :param feature_target: the target feature, which is being estimated.
        :param model_labals: the target labels (default [0, 1]).
        :param normalize: to normalise the labels.
        :param title: the figure title.
        :param cmap: the plot color.
        :return: the plot object, and the data used to plot.
        """
        summaries = dict()
        tick_marks = np.arange(len(model_labals))

        # Compute confusion matrix
        summaries["cnf_matrix"] = confusion_matrix(feature_target, predicted_scores)
        np.set_printoptions(precision=2)

        # plot metadata
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.clf()
        ax.title(title + ' Average Precision={0:0.2f}'.format(summaries["avg_precision"]))
        ax.ylabel('True label')
        ax.xlabel('Predicted label')
        ax.xticks(tick_marks, model_labals, rotation=45)
        ax.yticks(tick_marks, model_labals)
        ax.grid()
        ax.colorbar()

        # plot matrix
        plt.imshow(summaries["cnf_matrix"], interpolation='nearest', cmap=cmap)

        if normalize:
            summaries["cnf_matrix"] = \
                summaries["cnf_matrix"].astype('float') / summaries["cnf_matrix"].sum(axis=1)[:, np.newaxis]

        thresh = summaries["cnf_matrix"].max() / 2.
        for i, j in itertools.product(range(summaries["cnf_matrix"].shape[0]), range(summaries["cnf_matrix"].shape[1])):
            plt.text(j, i, summaries["cnf_matrix"][i, j],
                     horizontalalignment="center",
                     color="white" if summaries["cnf_matrix"][i, j] > thresh else "black")
        plt.tight_layout()
        return fig, summaries

    @staticmethod
    def stepwise_model(summaries: Dict,
                       title: str="Step-Wise Train & Test",
                       lw: int=2) -> MatplotlibFigure:
        """Plot a performance summary plot for the step-wise training and testing.
        :param summaries: the summary statistics which will be used for plotting.
        It must contain 'Train_Precision', 'Train_Recall', 'Train_ROC', 'Test_Precision', 'Test_Recall', and 'Test_ROC'
        for each training and testing step.
        :param title: the figure title.
        :param lw: the line-width.
        :return: the plot object.
        """
        # plot metadata
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.clf()
        plt.title(title)
        plt.ylim([0.0, 1.05])
        plt.xlabel('Number of Features')
        plt.ylabel('Summary Statistics')
        plt.grid()

        plt.plot(summaries["Step"], summaries["Train_Precision"], lw=lw, color='r', label='Train - Precision')
        plt.plot(summaries["Step"], summaries["Train_Recall"], lw=lw, color='g', label='Train - Recall')
        plt.plot(summaries["Step"], summaries["Train_ROC"], lw=lw, color='b', label='Train - ROC')

        plt.plot(summaries["Step"], summaries["Test_Precision"], lw=lw, color='brown', label='Test - Precision')
        plt.plot(summaries["Step"], summaries["Test_Recall"], lw=lw, color='orange', label='Test - Recall')
        plt.plot(summaries["Step"], summaries["Test_ROC"], lw=lw, color='pink', label='Test - ROC')
        plt.legend(loc="lower left")
        return fig

    @staticmethod
    def precision_recall(predicted_scores: List,
                         feature_target: List,
                         title: str="Precision-Recall Curve",
                         lw: int=2) -> [MatplotlibFigure, Dict]:
        """Plot the precision-recall curve.
        "The precision-recall plot is a model-wide measure for evaluating binary classifiers
        and closely related to the ROC plot."
        :param predicted_scores: the predicted Scores.
        :param feature_target: the target feature, which is being estimated.
        :param title: the figure title.
        :param lw: the line-width.
        :return: the plot object, and the data used to plot.
        """
        summaries = dict()

        # calculate
        summaries["precision"], summaries["recall"], _ = metrics.precision_recall_curve(
            feature_target, predicted_scores)

        # summaries
        summaries["avg_precision"] = metrics.average_precision_score(feature_target, predicted_scores)

        # plot metadata
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.clf()
        plt.title(title + ' Average Precision={0:0.2f}'.format(summaries["avg_precision"]))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid()

        plt.plot(summaries["precision"], summaries["recall"],
                 lw=lw, color='navy', label='Precision-Recall curve')
        plt.legend(loc="lower left")
        return fig, summaries

    @staticmethod
    def precision_recall_multiple(predicted_scores_list: List,
                                  feature_target_list: List,
                                  label_list: List,
                                  marker_list: List,
                                  linestyle_list: List,
                                  color_list: List,
                                  title: str="Precision-Recall Curve",
                                  lw: int=2,
                                  markersize: int=6,
                                  markevery: int=10000,
                                  legend_prop: int=2,
                                  legend_markerscale: int=2) -> [MatplotlibFigure, Dict]:
        """Plot the precision-recall curve.
        "The precision-recall plot is a model-wide measure for evaluating binary classifiers
        and closely related to the ROC plot."
        :param predicted_scores_list: the predicted Scores (one or multiple).
        :param feature_target_list: the target feature, which is being estimated (one or multiple).
        :param label_list: the line label (one or multiple).
        :param marker_list: the line marker (one or multiple).
        :param linestyle_list: the line style (one or multiple).
        :param color_list: the line color (one or multiple).
        :param title: the figure title.
        :param lw: the line-width.
        :param markersize: the marker size.
        :param markevery: to mark every x point.
        :param legend_prop: the legend proportion
        :param legend_markerscale: The legend's marker scale.
        :return: the plot object, and the data used to plot.
        """

        # calculate summaries
        summaries = [None] * len(predicted_scores_list)
        for i in range(len(predicted_scores_list)):
            summaries[i] = dict()
            summaries[i]["precision"], summaries[i]["recall"], _ = metrics.precision_recall_curve(
                feature_target_list[i], predicted_scores_list[i])
            summaries[i]["avg_precision"] = metrics.average_precision_score(
                feature_target_list[i], predicted_scores_list[i])

        # plot metadata
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(nrows=1, ncols=1)
        plt.clf()
        plt.title(title)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid()

        for i in range(len(predicted_scores_list)):
            plt.plot(summaries[i]["precision"], summaries[i]["recall"],
                     markersize=markersize,
                     marker=marker_list[i],
                     markevery=markevery,
                     linestyle=linestyle_list[i],
                     linewidth=lw,
                     color=color_list[i],
                     label='Avg. Precision (' + label_list[i] + ')={0:0.2f}'.format(summaries[i]["avg_precision"]))

        plt.legend(loc="lower left", prop={'size': legend_prop}, markerscale=legend_markerscale)
        return fig, summaries

    @staticmethod
    def roc(predicted_scores: List,
            feature_target: List,
            title: str="ROC Curve",
            lw: int=2) -> [MatplotlibFigure, Dict]:
        """Plot the Receiver Operating Characteristic (ROC)
        :param predicted_scores: the predicted Scores.
        :param feature_target: the target feature, which is being estimated.
        :param title: the figure title.
        :param lw: the line-width.
        :return: the plot object, and the data used to plot.
        """
        summaries = dict()

        # calculate
        summaries["fpr"], summaries["tpr"], _ = metrics.roc_curve(feature_target, predicted_scores)

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

        plt.plot(summaries["fpr"], summaries["tpr"], color='r',
                 lw=lw, label='ROC curve (area = %0.2f)' % summaries["roc_auc"])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.legend(loc="lower right")
        return fig, summaries

    @staticmethod
    def roc_multiple(predicted_scores_list: List,
                     feature_target_list: List,
                     label_list: List,
                     marker_list: List,
                     linestyle_list: List,
                     color_list: List,
                     title: str="ROC Curve",
                     lw: int=2,
                     markersize: int=6,
                     markevery: int=10000,
                     legend_prop: int=2,
                     legend_markerscale: int=2) -> [MatplotlibFigure, Dict]:
        """Plot the Receiver Operating Characteristic (ROC)
        :param predicted_scores_list: the predicted Scores (one or multiple).
        :param feature_target_list: the target feature, which is being estimated (one or multiple).
        :param label_list: the line label (one or multiple).
        :param marker_list: the line marker (one or multiple).
        :param linestyle_list: the line style (one or multiple).
        :param color_list: the line color (one or multiple).
        :param title: the figure title.
        :param lw: the line-width.
        :param markersize: the marker size.
        :param markevery: to mark every x point.
        :param legend_prop: the legend proportion
        :param legend_markerscale: the legend's marker scale.
        :return: the plot object, and the data used to plot.
        """

        # calculate summaries
        summaries = [None] * len(predicted_scores_list)
        for i in range(len(predicted_scores_list)):
            summaries[i] = dict()
            summaries[i]["fpr"], summaries[i]["tpr"], _ = metrics.roc_curve(
                feature_target_list[i], predicted_scores_list[i])
            summaries[i]["roc_auc"] = metrics.auc(
                summaries[i]["fpr"], summaries[i]["tpr"])

        # plot metadata
        plt.clf()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(nrows=1, ncols=1)
        plt.title(title)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid()

        for i in range(len(predicted_scores_list)):
            plt.plot(summaries[i]["fpr"], summaries[i]["tpr"],
                     markersize=markersize,
                     marker=marker_list[i],
                     markevery=markevery,
                     linestyle=linestyle_list[i],
                     linewidth=lw,
                     color=color_list[i],
                     label='AUC(' + label_list[i] + ')={0:0.2f}'.format(summaries[i]["roc_auc"]))

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.legend(loc="lower right", prop={'size': legend_prop}, markerscale=legend_markerscale)
        return fig, summaries

    @staticmethod
    def learning_curve(estimator: Any,
                       features_indep_df: PandasDataFrame,
                       feature_target: List,
                       title: str="Learning Curve",
                       ylim: List=None,
                       cv: int=None,
                       n_jobs: int=-1,
                       train_sizes: List=np.linspace(.1, 1.0, 5)) -> [MatplotlibFigure, Dict]:
        """Plot the learning curve.
        "A learning curve shows the validation and training score of an estimator for varying numbers of training
        samples. It is a tool to find out how much we benefit from adding more training data and whether the estimator
        suffers more from a variance error or a bias error. If both the validation score and the training score
        converge to a value that is too low with increasing size of the training set, we will not benefit much
        from more training data."
        :param estimator: the object type that implements the “fit” and “predict” methods.
        An object of that type which is cloned for each validation.
        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param title: the figure title.
        :param ylim: the y-limit for the axis.
        :param cv: the cross-validation splitting strategy (optional).
        :param n_jobs: the number of jobs to run in parallel (default -1).
        :param train_sizes: the size of the training samples for the learning curve.
        :return: the plot object, and the data used to plot.
        """
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

    @staticmethod
    def validation_curve(estimator: Any,
                         features_indep_df: PandasDataFrame,
                         feature_target: List,
                         param_name: str,
                         param_range: List,
                         title: str="Learning Curve",
                         ylim: List=None,
                         cv: int=None,
                         lw: int=2,
                         n_jobs: int=-1) -> [MatplotlibFigure, Dict]:
        """Plot the validation curve
        "it is sometimes helpful to plot the influence of a single hyperparameter on the training score and the
        validation score to find out whether the estimator is overfitting or underfitting for some hyperparameter
        values."
        :param estimator: the object type that implements the “fit” and “predict” methods.
        An object of that type which is cloned for each validation.
        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param param_name: the N=name of the parameter that will be varied.
        :param param_range: the values of the parameter that will be evaluated.
        :param title: the figure title.
        :param ylim: the y-limit for the axis.
        :param cv: the cross-validation splitting strategy (optional).
        :param lw: the line-width.
        :param n_jobs: the number of jobs to run in parallel (default -1).
        :return: the plot object, and the data used to plot.
        """
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

    @staticmethod
    def distribution_bar(feature: List,
                         feature_name: str,
                         title: str,
                         ylim: List=[0.0, 1.05]) -> tuple:
        """Plot distribution, using bar plot.
        :param feature: the value of the feature.
        :param feature_name: the name of the feature.
        :param title: the figure title.
        :param ylim: the y-limit for the axis.
        :return: the plot object.
        """
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

    @staticmethod
    def distribution_hist(feature: List,
                          feature_name: str,
                          title: str,
                          num_bins: int=50,
                          ylim: List=[0.0, 1.05]) -> tuple:
        """Plot distribution, using histogram.
        :param feature: the value of the feature.
        :param feature_name: the name of the feature.
        :param title: the figure title.
        :param num_bins: number of bins in the histogram.
        :param ylim: the y-limit for the axis.
        :return: the plot object.
        """
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

    @staticmethod
    def distribution_kde(feature: List,
                         feature_name: str,
                         title: str,
                         x_values: List=None,
                         kernel: str="gaussian",
                         bandwidth: float=0.5,
                         ylim: List=[0.0, 1.05]) -> List:
        """Plot distribution, using Kernel Density Estimation (KDE).
        :param feature: the value of the feature.
        :param feature_name: the name of the feature.
        :param title: the figure title.
        :param x_values: the grid to use for plotting (default: based on the feature range and size)
        :param kernel: the kernel to use. Valid kernels are
        :param bandwidth: the bandwidth of the kernel.
        :param ylim: the y-limit for the axis.
        :return: the plot object.
        """
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
