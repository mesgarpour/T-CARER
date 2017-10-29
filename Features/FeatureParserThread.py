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
""" It reads and parses the variables, then it generate features, in threaded batches.
"""

from typing import List, TypeVar, Dict
import numpy as np
import statistics
from scipy.stats import itemfreq

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


class FeatureParserThread:

    @staticmethod
    def aggregate_cell(postfixes: str,
                       variable_type: str,
                       prevalence: Dict,
                       variable_cell: str) -> NumpyNdarray:
        """Aggregate the variable value, based on the selected aggregated functions.
        :param postfixes: the aggregated variable.
        :param variable_type: the type of the input variable.
        :param prevalence: the prevalence dictionary of values for all the variables.
        :param variable_cell: the variable value (a single row) to aggregate.
        :return: the aggregated value (a single row).
        """
        features_temp = np.zeros([len(postfixes)])

        # if null or empty
        if variable_cell is None or variable_cell == "":
            return features_temp

        # parse variables
        # Note: replace None with '0'
        variable_cell = variable_cell.split('|')
        variable_cell = [v2 for v1 in variable_cell for v2 in set(v1.split(','))]
        variable_cell = [v if v != "" else 0 for v in variable_cell]
        if variable_type == "INT":
            variable_cell = list(map(int, variable_cell))

        # generate freq. table
        freq = itemfreq(variable_cell)
        freq = np.array([tuple(row) for row in freq if row[1] != 0], dtype=[('value', 'int'), ('freq', 'int')])
        freq_sorted = np.sort(freq, order=['freq', 'value'])[::-1]['value']
        freq_dic = dict(zip(map(str, freq['value']), freq['freq']))

        # set
        for p in range(len(postfixes)):
            if len(postfixes[p]) > 11 and postfixes[p][0:11] == "prevalence_":
                index = int(postfixes[p].split('_')[1]) - 1
                if index < len(prevalence):
                    value = prevalence[index]
                    if str(value) in freq_dic.keys():
                        features_temp[p] = freq_dic[value]
            elif len(postfixes[p]) > 9 and postfixes[p][0:9] == "max_freq_":
                index = int(postfixes[p][9:]) - 1
                if len(freq_sorted) > index:
                    features_temp[p] = freq_sorted[index]
            elif postfixes[p] == "others_cnt":
                features_temp[p] = len(freq_sorted)  # np.count_nonzero(variable_cell)
            elif postfixes[p] == "max":
                features_temp[p] = max(variable_cell)
            elif postfixes[p] == "avg":
                features_temp[p] = statistics.mean(variable_cell)
            elif postfixes[p] == "min":
                features_temp[p] = min(variable_cell)
            elif postfixes[p] == "median":
                features_temp[p] = statistics.median(variable_cell)
            else:
                raise ValueError(postfixes)
        return features_temp

    @staticmethod
    def prevalence_cell(variable_cell: str) -> List:
        """Parse the inputted variable value (a single row), to a list of value.
        :param variable_cell: the variable value (a single row), to calculate the prevalence.
        :return: the list of values of the current variable value.
        """
        # if null or empty
        if variable_cell is None or variable_cell == "":
            return []
        else:
            # parse variables
            # Note: replace None with '0'
            variable_cell = variable_cell.split('|')
            variable_cell = [v2 for v1 in variable_cell for v2 in set(v1.split(','))]
            variable_cell = [v if v != "" else 0 for v in variable_cell]
            variable_cell = list(map(str, variable_cell))
            return variable_cell
