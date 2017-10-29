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
""" It reads and parses the variables, then it generate features.
"""

from typing import List, TypeVar, Dict
import sys
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
from collections import Counter
from ReadersWriters.ReadersWriters import ReadersWriters
import logging
from Features.FeatureParserThread import FeatureParserThread
from Configs.CONSTANTS import CONSTANTS

PandasDataFrame = TypeVar('DataFrame')
NumpyNdarray = TypeVar('ndarray')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class FeatureParser:

    def __init__(self,
                 variables_settings: PandasDataFrame,
                 output_path: str,
                 output_table: str):
        """Initialise the objects and constants.
        :param variables_settings:
        :param output_path: the output path.
        :param output_table: the output table name.
        """
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__variables_settings = variables_settings
        self.__output_path = output_path
        self.__output_table = output_table
        self.__readers_writers = ReadersWriters()
        self.__FeatureParserThread = FeatureParserThread()

    def generate(self,
                 history_table: str,
                 features: PandasDataFrame,
                 variables: PandasDataFrame,
                 prevalence: Dict) -> PandasDataFrame:
        """

        :param history_table: the source table alias name (a.k.a. history table name) that features belong to
            (e.g. inpatient, or outpatient).
        :param features: the output features.
        :param variables: the input variables.
        :param prevalence: the prevalence dictionary of values for all the variables.
        :return: the output features.
        """
        variables_settings = self.__variables_settings[self.__variables_settings["Table_History_Name"] == history_table]

        for _, row in variables_settings.iterrows():
            self.__logger.info("variable: " + row["Variable_Name"] + " ...")

            if not pd.isnull(row["Variable_Aggregation"]):
                postfixes = row["Variable_Aggregation"].replace(' ', '').split(',')
                # aggregate stats
                features_temp = self.__aggregate(
                    variables[row["Variable_Name"]], row["Variable_Type_Original"],
                    postfixes, prevalence[row["Variable_Name"]])
                for p in range(len(postfixes)):
                    # feature name
                    feature_name = row["Variable_Name"] + "_" + postfixes[p]
                    # set
                    features[feature_name] = features_temp[:, p]
            else:
                # init and replace none by zero
                features_temp = np.nan_to_num(variables[row["Variable_Name"]])
                features_temp = np.where(features_temp == np.array(None), 0, features_temp)
                # set
                features[row["Variable_Name"]] = features_temp
        return features

    def __aggregate(self,
                    variable: PandasDataFrame,
                    variable_type: str,
                    postfixes: str,
                    prevalence: Dict) -> NumpyNdarray:
        """

        :param variable: the input variable.
        :param variable_type: the type of input variable.
        :param postfixes: name of the aggregation functions.
        :param prevalence: the prevalence dictionary of values for all the variables.
        :return: the aggregated variable.
        """
        try:
            with mp.Pool() as pool:
                features_temp = pool.map(
                    partial(self.__FeatureParserThread.aggregate_cell, postfixes, variable_type, prevalence), variable)
        except ValueError as exception:
            self.__logger.error(__name__ + " - Invalid configuration(s): " + str(exception))
            sys.exit()

        features_temp = np.asarray(features_temp)
        return features_temp

    def prevalence(self,
                   variable: PandasDataFrame,
                   variable_name: str) -> List:
        """
        :param variable: the input variable.
        :param variable_name: the name of the input variable.
        :return: the prevalence of values for all the variables.
        """
        try:
            with mp.Pool() as pool:
                prevalence_temp = pool.map(
                    partial(self.__FeatureParserThread.prevalence_cell), variable)
        except ValueError as exception:
            self.__logger.error(__name__ + " - Invalid configuration(s): " + str(exception))
            sys.exit()

        prevalence_temp = [sub2 for sub1 in prevalence_temp for sub2 in sub1]
        prevalence = Counter(prevalence_temp).most_common()
        self.__readers_writers.save_text(self.__output_path, self.__output_table,
                                         [variable_name, '; '.join([str(p[0]) + ":" + str(p[1]) for p in prevalence])],
                                         append=True, ext="txt")
        prevalence = [p[0] for p in prevalence]
        return prevalence
