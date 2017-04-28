#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sklearn import ensemble
from sklearn import feature_selection
from sklearn import tree
from sklearn import svm
from sklearn.svm import SVR
from sklearn.linear_model import RandomizedLogisticRegression
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


class FeatureSelection:
    def __init__(self):
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)

    def rank_random_forest_breiman(self, features_indep, feature_target, **kwargs):
        """random forest classifier (Brieman)
        model.estimators_
        model.classes_
        model.n_classes_
        model.n_features_
        model.n_outputs_
        model.feature_importances_

        n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True,
        oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None
        """
        self.__logger.debug(__name__)
        self.__logger.info("Running Random Forest Classifier (Brieman)")

        classifier = ensemble.RandomForestClassifier(**kwargs)
        model = classifier.fit(features_indep, feature_target)
        return model

    def rank_random_logistic_regression(self, features_indep, feature_target, **kwargs):
        """random forest classifier (Brieman)
        model.estimators_
        model.classes_
        model.n_classes_
        model.n_features_
        model.n_outputs_
        model.feature_importances_

        C=1, scaling=0.5, sample_fraction=0.75, n_resampling=200, selection_threshold=0.25, tol=0.001,
        fit_intercept=True, verbose=False, normalize=True, random_state=None, n_jobs=1, pre_dispatch='3*n_jobs'
        """
        self.__logger.debug(__name__)
        self.__logger.info("Running Random Logistic Regression")

        classifier = RandomizedLogisticRegression(**kwargs)
        model = classifier.fit(features_indep, feature_target)
        return model

    def rank_svm_c_support(self, features_indep, feature_target, **kwargs):
        """C-Support Vector Classification
        model.support_
        model.support_vectors_
        model.n_support_
        model.dual_coef_
        model.coef_
        model.intercept_

        C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
        tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,
        random_state=None, sample_weight=None
        """
        self.__logger.debug(__name__)
        self.__logger.info("Running C-Support Vector Classification")

        classifier = svm.SVC(**kwargs)
        model = classifier.fit(features_indep, feature_target)
        return model

    def rank_tree_brieman(self, features_indep, feature_target, **kwargs):
        """decision tree classifier (Brieman)
        model.classes_
        model.feature_importances_
        model.max_features_
        model.n_classes_
        model.n_features_
        model.n_outputs_
        model.tree_

        criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None,
        presort=True, sample_weight=None, check_input=True, X_idx_sorted=None
        """
        self.__logger.debug(__name__)
        self.__logger.info("Running Decision Tree Classifier (Brieman)")

        classifier = tree.DecisionTreeClassifier(**kwargs)
        model = classifier.fit(features_indep, feature_target)
        return model

    def rank_tree_gbrt(self, features_indep, feature_target, **kwargs):
        """Gradient Boosted Regression Trees (GBRT)
        model.feature_importances_
        model.train_score_
        model.loss_
        model.init
        model.estimators_

        loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, alpha=0.9,
        verbose=0, max_leaf_nodes=None, warm_start=False, presort=True
        """
        self.__logger.debug(__name__)
        self.__logger.info("Running Gradient Boosted Regression Trees (GBRT)")

        classifier = ensemble.GradientBoostingRegressor(**kwargs)
        model = classifier.fit(features_indep, feature_target)
        return model

    def selector_logistic_rfe(self, features_indep, feature_target, kernel="linear"):
        """Feature ranking with recursive feature elimination
        model.n_features_
        model.support_   # selected features
        model.ranking_
        model.grid_scores_
        model.estimator_
        """
        self.__logger.debug(__name__)
        self.__logger.info("Running Feature Ranking with Recursive Feature Elimination")

        estimator = SVR(kernel=kernel)
        selector = feature_selection.RFECV(
            estimator=estimator, step=1, cv=None, scoring=None, verbose=0)
        model = selector.fit(features_indep, feature_target)
        return model

    def selector_univarite_selection_kbest_chi2(self, features_indep, feature_target, kbest):
        return self.__selector_univarite_selection_kbest(features_indep, feature_target,
                                                         feature_selection.chi2, kbest)

    def selector_univarite_selection_kbest_f_classif(self, features_indep, feature_target, kbest):
        return self.__selector_univarite_selection_kbest(features_indep, feature_target,
                                                         feature_selection.f_classif, kbest)

    def __selector_univarite_selection_kbest(self, features_indep, feature_target, score_func, kbest):
        """Univariate feature selection with configurable strategy
        model.scores_
        model.pvalues_
        """
        self.__logger.debug(__name__)
        self.__logger.info("Running Univariate Feature Selection with Configurable Strategy")
        kbest = int(float(kbest) * features_indep.shape[1])

        selector = feature_selection.SelectKBest(
            score_func=score_func, k=kbest)
        model = selector.fit(features_indep, feature_target)
        return model
