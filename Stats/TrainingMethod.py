#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from Stats._LogisticRegression import _LogisticRegression
from Stats._LogisticRegressionCV import _LogisticRegressionCV
from Stats._MixedLinearModel import _MixedLinearModel
from Stats._RandomForestClassifier import _RandomForestClassifier
from Stats._GradientBoostingClassifier import _GradientBoostingClassifier
from Stats._DecisionTreeClassifier import _DecisionTreeClassifier
from Stats._KNeighborsClassifier import _KNeighborsClassifier
from Stats._NaiveBayes import _NaiveBayes
from Stats._NeuralNetwork import _NeuralNetwork
from ReadersWrites.ReadersWriters import ReadersWriters
from Configs.CONSTANTS import CONSTANTS
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import sys
import logging

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class TrainingMethod:
    def __init__(self, method_name, path=None, title=None):
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

    def __init__method(self, method_name, model_labels=None, model_train=None,
                       model_predict=dict(), model_cross_validate=None):
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
        self.model_predict = model_predict
        self.model_cross_validate = model_cross_validate

    def train(self, features_indep_df, feature_target, **kwargs):
        self.__logger.debug(__name__)
        self.__logger.info("Training")
        self.model_labels = list(features_indep_df.columns.values)
        self.model_train = self.__method.train(
            features_indep_df[self.model_labels], feature_target, self.model_labels, **kwargs)
        return self.model_train

    def train_summaries(self):
        return self.__method.train_summaries(self.model_train)

    def predict(self, features_indep_df, sample_name):
        self.__logger.debug(__name__)
        self.__logger.info("Predicting")
        self.model_predict[sample_name] = self.__method.predict(self.model_train, features_indep_df[self.model_labels])
        return self.model_predict[sample_name]

    def predict_summaries(self, feature_target, sample_name):
        return self.__method.predict_summaries(self.model_predict[sample_name], feature_target)

    def predict_summaries_risk_bands(self, feature_target, sample_name, cutoffs=np.arange(0, 1.05, 0.05)):
        return self.__method.predict_summaries_cutoffs_table(
            self.model_predict[sample_name]['score'], feature_target, cutoffs)

    def cross_validate(self, features_indep_df, feature_target,
                       scoring="neg_mean_squared_error", cv=10):
        self.__logger.debug(__name__)
        self.__logger.info("Cross-Validate")

        self.model_cross_validate = cross_val_score(
            self.model_train, features_indep_df[self.model_labels], feature_target, scoring=scoring, cv=cv)
        return self.model_cross_validate

    def cross_validate_summaries(self):
        return self.model_cross_validate

    def save_model(self, path, title):
        self.__logger.debug(__name__)
        self.__logger.info("Saving model")
        objects = dict()
        objects['method_name'] = self.method_name
        objects['model_labels'] = self.model_labels
        objects['model_train'] = self.model_train
        objects['model_predict'] = self.model_predict
        objects['model_cross_validate'] = self.model_cross_validate
        self.__readers_writers.save_serialised(path, title, objects=objects)

    def save_predictions(self, feature_id_label, feature_id, feature_pred, feature_target,
                         feature_poly_label, feature_poly, schema, table, query_batch_size, overwriting):
        self.__logger.debug(__name__)
        self.__logger.info("Saving predictions")
        data = pd.DataFrame({feature_id_label: np.array(feature_id, dtype=np.int32),
                             'prediction': np.array(feature_pred, dtype=np.float16),
                             'label': np.array(feature_target, dtype=np.int8),
                             feature_poly_label: np.array(feature_poly, dtype=np.float16)})
        if feature_poly_label is None:
            data.drop(feature_poly_label, axis=1)

        # overwriting
        exists = self.__readers_writers.exists_mysql(schema=schema, table=table)
        if exists is True:
            if overwriting is False:
                response = self.__readers_writers.question_overwrite(
                    "the overwrite of predictions to the following table: " + schema + "." + table)
                if response is False:
                    self.__logger.error(__name__ + " - " + "Overwrite is cancelled")
                    sys.exit()

            # drop table
            query_drop = self.__readers_writers.mysql_query_drop(table)
            self.__readers_writers.save_mysql(query=query_drop, data=None, schema=schema, table=None)

        # save
        self.__readers_writers.save_mysql(query=None, data=data, schema=schema, table=table, batch=query_batch_size)

    def load(self, path, title):
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
