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
""" It applies feature factorisation (categorisation) using independent threads.
"""

from typing import Dict, TypeVar
from sklearn import preprocessing
import pandas as pd

PandasDataFrame = TypeVar('DataFrame')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class FactoringThread:
    def __init__(self,
                 df: PandasDataFrame,
                 categories_dic: Dict,
                 labels_dic: Dict):
        """Initialise the objects and constants.
        :param df: the inputted dataframe to process.
        :param categories_dic: the categorisation dictionary.
        :param labels_dic: the name of the new features.
        """
        self.__df = df
        self.__categories_dic = categories_dic
        self.__labels_dic = labels_dic

    def factor_arr_multiple(self,
                            label_group: str) -> PandasDataFrame:
        """Categorise multiple features.
        :param label_group: the names of features to be categorised.
        :return: the categorised features.
        """
        labels_encoded = list(self.__categories_dic[label_group].keys())
        df_encoded = self.__factor_arr(labels_encoded[0], label_group)

        if len(labels_encoded) > 1:
            for label in labels_encoded[1:]:
                df_encoded = df_encoded.add(self.__factor_arr(label, label_group))
        return df_encoded

    def factor_arr(self,
                   label: str) -> PandasDataFrame:
        """Categorise a single feature.
        :param label: the name of the feature to be categorised.
        :return: the categorised feature.
        """
        df_encoded = preprocessing.label_binarize(self.__df[label], classes=self.__categories_dic[label])
        df_encoded = pd.DataFrame(df_encoded, columns=self.__labels_dic[label])
        return df_encoded

    def __factor_arr(self,
                     label: str,
                     label_group: str) -> PandasDataFrame:
        """Categorise a list using the 'preprocessing.label_binarize'.
        :param label: the name of the feature to be categorised.
        :param label_group: the name of the feature group in the categorisation dictionary.
        :return: the categorised feature.
        """
        df_encoded = preprocessing.label_binarize(self.__df[label], classes=self.__categories_dic[label_group][label])
        df_encoded = pd.DataFrame(df_encoded, columns=self.__labels_dic[label_group])
        return df_encoded
