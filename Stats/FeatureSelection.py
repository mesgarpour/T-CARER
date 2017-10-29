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
"""It is an interface for ranking features importance.
"""

from typing import List, TypeVar, Any
from sklearn import ensemble
from sklearn import feature_selection
from sklearn import tree
from sklearn import svm
from sklearn.svm import SVR
from sklearn.linear_model import RandomizedLogisticRegression
import logging
from Configs.CONSTANTS import CONSTANTS

PandasDataFrame = TypeVar('DataFrame')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class FeatureSelection:
    def __init__(self):
        """Initialise the objects and constants.
        """
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)

    def rank_random_forest_breiman(self,
                                   features_indep_df: PandasDataFrame,
                                   feature_target: List,
                                   n_jobs: int=-1,
                                   **kwargs: Any) -> object:
        """Use Brieman Random Forest Classifier to rank features.
        Attributes:
        model.estimators_
        model.classes_
        model.n_classes_
        model.n_features_
        model.n_outputs_
        model.feature_importances_

        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param n_jobs: number of CPUs to use during the resampling. If ‘-1’, use all the CPUs.
        :param kwargs: n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True,
        oob_score=False, random_state=None, verbose=0, warm_start=False, class_weight=None
        :return: the importance ranking model.
        """
        self.__logger.debug("Run Random Forest Classifier (Brieman).")
        classifier = ensemble.RandomForestClassifier(n_jobs=n_jobs, **kwargs)
        return classifier.fit(features_indep_df, feature_target)

    def rank_random_logistic_regression(self,
                                        features_indep_df: PandasDataFrame,
                                        feature_target: List,
                                        n_jobs: int=-1,
                                        **kwargs: Any) -> object:
        """Use Randomized Logistic Regression to rank features.
        Attributes:
        model.scores_
        model.all_scores_

        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param n_jobs: number of CPUs to use during the resampling. If ‘-1’, use all the CPUs.
        :param kwargs: C=1, scaling=0.5, sample_fraction=0.75, n_resampling=200, selection_threshold=0.25, tol=0.001,
        fit_intercept=True, verbose=False, normalize=True, random_state=None, pre_dispatch='3*n_jobs'
        :return: the importance ranking model.
        """
        self.__logger.debug("Run Random Logistic Regression.")
        classifier = RandomizedLogisticRegression(n_jobs=n_jobs, **kwargs)
        return classifier.fit(features_indep_df, feature_target)

    def rank_svm_c_support(self,
                           features_indep_df: PandasDataFrame,
                           feature_target: List,
                           **kwargs: Any) -> object:
        """Use Scalable Linear Support Vector Machine for classification.
        In C-Support Vector Classification (SVC), the C parameter trades off misclassification of training examples
        against simplicity of the decision surface.
        Attributes:
        model.support_
        model.support_vectors_
        model.n_support_
        model.dual_coef_
        model.coef_
        model.intercept_

        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param kwargs: C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
        tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,
        random_state=None
        :return: the importance ranking model.
        """
        self.__logger.debug("Run C-Support Vector Classification.")
        classifier = svm.SVC(**kwargs)
        return classifier.fit(features_indep_df, feature_target)

    def rank_tree_brieman(self,
                          features_indep_df: PandasDataFrame,
                          feature_target: List,
                          **kwargs: Any) -> object:
        """Use Brieman decision tree classifier to rank features.
        Attributes:
        model.classes_
        model.feature_importances_
        model.max_features_
        model.n_classes_
        model.n_features_
        model.n_outputs_
        model.tree_

        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param kwargs: criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
        min_impurity_split=1e-07, class_weight=None, presort=False
        :return: the importance ranking model.
        """
        self.__logger.debug("Run Decision Tree Classifier (Brieman).")
        classifier = tree.DecisionTreeClassifier(**kwargs)
        return classifier.fit(features_indep_df, feature_target)

    def rank_tree_gbrt(self,
                       features_indep_df: PandasDataFrame,
                       feature_target: List,
                       **kwargs: Any) -> object:
        """Use Gradient Boosted Regression Trees (GBRT) to rank features.
        Attributes:
        model.feature_importances_
        model.train_score_
        model.loss_
        model.init
        model.estimators_

        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param kwargs: loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07,
        init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False,
        presort='auto'
        :return: the importance ranking model.
        """
        self.__logger.debug("Run Gradient Boosted Regression Trees (GBRT).")
        classifier = ensemble.GradientBoostingRegressor(**kwargs)
        return classifier.fit(features_indep_df, feature_target)

    def selector_logistic_rfe(self,
                              features_indep_df: PandasDataFrame,
                              feature_target: List,
                              kernel: str="linear",
                              n_jobs: int=-1,
                              **kwargs: Any) -> object:
        """Select top features using recursive feature elimination and cross-validated selection of the best number
        of features, to rank features.
        Attributes:
        model.n_features_
        model.support_
        model.ranking_
        model.grid_scores_
        model.estimator_

        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param kernel: Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’,
        ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used.
        :param n_jobs: number of CPUs to use during the resampling. If ‘-1’, use all the CPUs.
        :param kwargs: step=1, cv=None, scoring=None, verbose=0
        :return: the feature selection model.
        """
        self.__logger.debug("Run Feature Ranking with Recursive Feature Elimination.")
        estimator = SVR(kernel=kernel)
        selector = feature_selection.RFECV(estimator=estimator, n_jobs=n_jobs, **kwargs)
        return selector.fit(features_indep_df, feature_target)

    def selector_univarite_selection_kbest_chi2(self,
                                                features_indep_df: PandasDataFrame,
                                                feature_target: List,
                                                kbest: int) -> object:
        """Select features according to the k highest scores, using 'chi2':
        Chi-squared stats of non-negative features for classification tasks.
        Attributes:
        model.scores_
        model.pvalues_

        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param kbest: number of top features to select. The “all” option bypasses selection, for use in a parameter
        search.
        :return: the feature selection model.
        """
        self.__logger.debug("Select features according to the k highest scores, using 'chi2'.")
        return self.__selector_univarite_selection_kbest(features_indep_df,
                                                         feature_target,
                                                         feature_selection.chi2,
                                                         kbest)

    def selector_univarite_selection_kbest_f_classif(self,
                                                     features_indep_df: PandasDataFrame,
                                                     feature_target: List,
                                                     kbest: int) -> object:
        """Select features according to the k highest scores, using 'f_classif':
        ANOVA F-value between label/feature for classification tasks.
        Attributes:
        model.scores_
        model.pvalues_

        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param kbest: number of top features to select. The “all” option bypasses selection, for use in a parameter
        search.
        :return: the feature selection model.
        """
        self.__logger.debug("Select features according to the k highest scores, using 'f_classif'.")
        return self.__selector_univarite_selection_kbest(features_indep_df, feature_target,
                                                         feature_selection.f_classif, kbest)

    def __selector_univarite_selection_kbest(self,
                                             features_indep_df: PandasDataFrame,
                                             feature_target: List,
                                             score_func: Any,
                                             kbest: int) -> object:
        """Select features according to the k highest scores.
        Attributes:
        model.scores_
        model.pvalues_

        :param features_indep_df: the independent features, which are inputted into the model.
        :param feature_target: the target feature, which is being estimated.
        :param score_func: Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or
        a single array with scores.
        :param kbest: number of top features to select. The “all” option bypasses selection, for use in a parameter
        search.
        :return: the feature selection model.
        """
        self.__logger.debug("Run Univariate Feature Selection with Configurable Strategy.")
        kbest = int(float(kbest) * features_indep_df.shape[1])
        selector = feature_selection.SelectKBest(
            score_func=score_func, k=kbest)
        return selector.fit(features_indep_df, feature_target)

