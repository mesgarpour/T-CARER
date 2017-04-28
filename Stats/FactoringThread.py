#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sklearn import preprocessing
import pandas as pd

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class FactoringThread:
    def __init__(self, df, categories_dic, labels_dic):
        self.__df = df
        self.__categories_dic = categories_dic
        self.__labels_dic = labels_dic

    def factor_arr_group(self, label_group):
        labels_encoded = list(self.__categories_dic[label_group].keys())
        df_encoded = self.__factor_arr(labels_encoded[0], label_group)

        if len(labels_encoded) > 1:
            for label in labels_encoded[1:]:
                df_encoded = df_encoded.add(self.__factor_arr(label, label_group))
        return df_encoded

    def factor_arr(self, label):
        df_encoded = preprocessing.label_binarize(self.__df[label], classes=self.__categories_dic[label])
        df_encoded = pd.DataFrame(df_encoded, columns=self.__labels_dic[label])
        return df_encoded

    def __factor_arr(self, label, label_group):
        df_encoded = preprocessing.label_binarize(self.__df[label], classes=self.__categories_dic[label_group][label])
        df_encoded = pd.DataFrame(df_encoded, columns=self.__labels_dic[label_group])
        return df_encoded
