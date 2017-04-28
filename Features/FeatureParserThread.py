#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import statistics
from scipy.stats import itemfreq

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class FeatureParserThread:

    @staticmethod
    def aggregate_cell(postfixes, variable_type, prevalence, variable_cell):
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
    def prevalence_cell(variable_cell):
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
