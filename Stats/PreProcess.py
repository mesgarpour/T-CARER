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
"""It is an interface for the developed pre-processing functions (factoring and near-zero-variance,
high-linear-correlation) and statistical summaries.
"""

from typing import Dict, List, TypeVar, Any
from Configs.CONSTANTS import CONSTANTS
from ReadersWriters.PyConfigParser import PyConfigParser
from ReadersWriters.ReadersWriters import ReadersWriters
from Stats.FactoringThread import FactoringThread
from Stats.TransformThread import TransformThread
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import OrderedDict
from sklearn import feature_selection
from scipy.stats import stats
from functools import partial
import logging

PandasDataFrame = TypeVar('DataFrame')
CollectionsOrderedDict = TypeVar('OrderedDict')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class PreProcess:
    def __init__(self,
                 output_path: str):
        """Initialise the objects and constants.
        :param output_path:
        """
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__output_path = output_path
        self.__readers_writers = ReadersWriters()

    def stats_discrete_df(self,
                          df: PandasDataFrame,
                          includes: List,
                          file_name: str) -> PandasDataFrame:
        """Calculate the odds ratio for all the features that are included and all the categorical states.
        :param df: the features dataframe.
        :param includes: the name of included features.
        :param file_name: the name of the summary output file.
        :return: the summary output.
        """
        self.__logger.debug("Produce statistics for discrete features.")
        summaries = None
        self.__readers_writers.save_csv(path=self.__output_path,
                                        title=file_name,
                                        data=[],
                                        append=False)

        for f_name in includes:
            if f_name in df:
                self.__readers_writers.save_csv(path=self.__output_path,
                                                title=file_name,
                                                data=["Feature Name", f_name],
                                                append=True)
                summaries = stats.itemfreq(df[f_name])
                summaries = pd.DataFrame({"value": summaries[:, 0], "freq": summaries[:, 1]})
                summaries = summaries.sort_values("freq", ascending=False)
                self.__readers_writers.save_csv(path=self.__output_path,
                                                title=file_name,
                                                data=summaries,
                                                append=True,
                                                header=True)
        return summaries

    def stats_continuous_df(self,
                            df: PandasDataFrame,
                            includes: List,
                            file_name: str) -> PandasDataFrame:
        """Calculate the descriptive statistics for all the included continuous features.
        :param df: the features dataframe.
        :param includes: the name of included features.
        :param file_name: the name of the summary output file.
        :return: the summary output.
        """
        self.__logger.debug("Produce statistics for continuous features.")
        summaries = None
        self.__readers_writers.save_csv(path=self.__output_path,
                                        title=file_name,
                                        data=[],
                                        append=False)

        for f_name in includes:
            if f_name in df:
                self.__readers_writers.save_csv(path=self.__output_path,
                                                title=file_name,
                                                data=["Feature Name", f_name],
                                                append=True)
                summaries = df[f_name].apply(pd.to_numeric).describe(
                    percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).transpose()
                summaries = pd.Series.to_frame(summaries).transpose()
                self.__readers_writers.save_csv(path=self.__output_path,
                                                title=file_name,
                                                data=summaries,
                                                append=True,
                                                header=True)
        return summaries

    def factoring_group_wise(self,
                             df: PandasDataFrame,
                             categories_dic: Dict,
                             labels_dic: Dict,
                             dtypes_dic: Dict,
                             threaded: bool=False) -> PandasDataFrame:
        """Categorise groups of features that are selected.
        :param df: the features dataframe.
        :param categories_dic: the dictionary of the categorical states for the included features.
        :param labels_dic: the dictionary of the features names of the categorised features.
        :param dtypes_dic: the dictionary of the dtypes of the categorised features.
        :param threaded: indicates if it is multi-threaded.
        :return: the inputted dataframe with categorised features (if applicable).
        """
        self.__logger.debug("Categorise groups of features.")
        categories_dic = OrderedDict(categories_dic)

        if threaded is not True:
            pool_df_encoded = self.__factoring_group_wise_series(df, categories_dic, labels_dic)
        else:
            pool_df_encoded = self.__factoring_group_wise_threaded(df, categories_dic, labels_dic)

        # encoded labels
        labels_encoded = []
        for label_group in categories_dic.keys():
            labels_encoded += list(categories_dic[label_group].keys())

        # preserve types
        dtype_orig = {**df.dtypes.to_dict(), **dtypes_dic}
        dtype_orig = pd.DataFrame(dtype_orig, index=[0]).dtypes
        for label in labels_encoded:
            del dtype_orig[label]

        # combine
        df = df.drop(labels_encoded, axis=1)
        df = pd.concat([df] + pool_df_encoded, axis=1)
        df = df.astype(dtype_orig)
        return df

    def __factoring_group_wise_series(self,
                                      df: PandasDataFrame,
                                      categories_dic: Dict,
                                      labels_dic: Dict) -> List:
        """Categorise a group of features that are selected (single-threaded).
        :param df: the features dataframe.
        :param categories_dic: the dictionary of the categorical states for the included features.
        :param labels_dic: the dictionary of the features names of the categorised features.
        :return: the categorised features.
        """
        self.__logger.debug("Categorise groups of features (single-threaded).")
        factoring_thread = FactoringThread(df, categories_dic, labels_dic)
        pool_df_encoded = []

        try:
            for label_group in categories_dic.keys():
                pool_df_encoded.append(factoring_thread.factor_arr_multiple(label_group))
        except ValueError as exception:
            self.__logger.error(__name__ + " - Invalid configuration(s): " + str(exception))
            sys.exit()
        return pool_df_encoded

    def __factoring_group_wise_threaded(self,
                                        df: PandasDataFrame,
                                        categories_dic: Dict,
                                        labels_dic: Dict) -> List:
        """Categorise a group of features that are selected (multi-threaded).
        :param df: the features dataframe.
        :param categories_dic: the dictionary of the categorical states for the included features.
        :param labels_dic: the dictionary of the features names of the categorised features.
        :return: the categorised features.
        """
        self.__logger.debug("Categorise groups of features (multi-threaded).")
        factoring_thread = FactoringThread(df, categories_dic, labels_dic)
        try:
            with mp.Pool(processes=(mp.cpu_count() - 1)) as pool:
                pool_df_encoded = pool.map(
                    partial(factoring_thread.factor_arr_multiple), categories_dic.keys())
        except ValueError as exception:
            self.__logger.error(__name__ + " - Invalid configuration(s): " + str(exception))
            sys.exit()
        return pool_df_encoded

    def factoring_feature_wise(self,
                               df: PandasDataFrame,
                               categories_dic: Dict,
                               labels_dic: Dict,
                               dtypes_dic: Dict,
                               threaded: bool=False) -> PandasDataFrame:
        """Categorise features that are selected.
        :param df: the features dataframe.
        :param categories_dic: the dictionary of the categorical states for the included features.
        :param labels_dic: the dictionary of the features names of the categorised features.
        :param dtypes_dic: the dictionary of the dtypes of the categorised features.
        :param threaded: indicates if it is multi-threaded.
        :return: the inputted dataframe with categorised features (if applicable).
        """
        self.__logger.debug("Categorise.")
        categories_dic = OrderedDict(categories_dic)

        if threaded is not True:
            pool_df_encoded = self.__factoring_feature_wise_series(df, categories_dic, labels_dic)
        else:
            pool_df_encoded = self.__factoring_feature_wise_threaded(df, categories_dic, labels_dic)

        # encoded labels
        labels_encoded = list(categories_dic.keys())

        # preserve types
        dtype_orig = {**df.dtypes.to_dict(), **dtypes_dic}
        dtype_orig = pd.DataFrame(dtype_orig, index=[0]).dtypes
        for label in labels_encoded:
            del dtype_orig[label]

        # combine
        df = df.drop(labels_encoded, axis=1)
        df = pd.concat([df] + pool_df_encoded, axis=1)
        df = df.astype(dtype_orig)
        return df

    def __factoring_feature_wise_series(self,
                                        df: PandasDataFrame,
                                        categories_dic: Dict,
                                        labels_dic: Dict) -> List:
        """Categorise features that are selected (single-threaded).
        :param df: the features dataframe.
        :param categories_dic: the dictionary of the categorical states for the included features.
        :param labels_dic: the dictionary of the features names of the categorised features.
        :return: the categorised features.
        """
        self.__logger.debug("Categorise (single-threaded).")
        factoring_thread = FactoringThread(df, categories_dic, labels_dic)
        pool_df_encoded = []

        try:
            for label_group in categories_dic.keys():
                pool_df_encoded.append(factoring_thread.factor_arr(label_group))
        except ValueError as exception:
            self.__logger.error(__name__ + " - Invalid configuration(s): " + str(exception))
            sys.exit()
        return pool_df_encoded

    def __factoring_feature_wise_threaded(self,
                                          df: PandasDataFrame,
                                          categories_dic: Dict,
                                          labels_dic: Dict) -> List:
        """Categorise features that are selected (multi-threaded).
        :param df: the features dataframe.
        :param categories_dic: the dictionary of the categorical states for the included features.
        :param labels_dic: the dictionary of the features names of the categorised features.
        :return: the categorised features.
        """
        self.__logger.debug("Categorise (multi-threaded).")
        factoring_thread = FactoringThread(df, categories_dic, labels_dic)
        try:
            with mp.Pool() as pool:
                pool_df_encoded = pool.map(
                    partial(factoring_thread.factor_arr), categories_dic.keys())
        except ValueError as exception:
            self.__logger.error(__name__ + " - Invalid configuration(s): " + str(exception))
            sys.exit()
        return pool_df_encoded

    def transform_df(self,
                     df: PandasDataFrame,
                     excludes: List,
                     transform_type: str,
                     threaded: bool=False,
                     method_args: Dict=None,
                     **kwargs: Any) -> [PandasDataFrame, Dict]:
        """Transform the included features, using the selected and configured method.
        :param df: the features dataframe.
        :param excludes: the name of excluded features.
        :param transform_type: the transformation type (options: 'scale', 'robust_scale', 'max_abs_scalar',
        'normalizer', 'kernel_centerer', 'yeo_johnson', 'box_cox')
        :param threaded: indicates if it is multi-threaded.
        :param method_args: the transformation arguments, which needs to preserved if it is applied to more than
        one data set.
        :param kwargs: the input argument for the selected transformation function.
        :return: the inputted dataframe with transformed features (if applicable).
        """
        self.__logger.info("Transform Features.")
        excludes = set(excludes)
        includes = [label for label in df.columns.values if label not in excludes]
        method_args = dict() if method_args is None else method_args

        # preserve types
        dtype_orig = df.dtypes.to_dict()
        for label in includes:
            dtype_orig[label] = 'f8'
        dtype_orig = pd.DataFrame(dtype_orig, index=[0]).dtypes
        df = df.astype(dtype_orig)

        # transform
        if threaded is False:
            df, method_args = self.__transform_df_series(df, includes, transform_type, **kwargs)
        else:
            df, method_args = self.__transform_df_threaded(df, includes, transform_type, method_args, **kwargs)
        return df, method_args

    def __transform_df_series(self,
                              df: PandasDataFrame,
                              includes: List,
                              transform_type: str,
                              method_args: Dict=None,
                              **kwargs: Any) -> [PandasDataFrame, Dict]:
        """Transform the included features, using the selected and configured method (single-threaded).
        :param df: the features dataframe.
        :param includes: the name of included features.
        :param transform_type: the transformation type (options: 'scale', 'robust_scale', 'max_abs_scalar',
        'normalizer', 'kernel_centerer', 'yeo_johnson', 'box_cox')
        :param method_args: the transformation arguments, which needs to preserved if it is applied to more than
        one data set.
        :param kwargs: the input argument for the selected transformation function.
        :return: the transformed feature.
        """
        self.__logger.debug("Transform features (single-threaded).")
        transform_thread = TransformThread(**kwargs)
        method_args = dict() if method_args is None else method_args

        try:
            if transform_type == "scale":
                for name in includes:
                    transform_thread.transform_scale_arr(df, method_args, name)
            elif transform_type == "robust_scale":
                for name in includes:
                    transform_thread.transform_robust_scale_arr(df, method_args, name)
            elif transform_type == "max_abs_scalar":
                for name in includes:
                    transform_thread.transform_max_abs_scalar_arr(df, method_args, name)
            elif transform_type == "normalizer":
                for name in includes:
                    transform_thread.transform_normalizer_arr(df, method_args, name)
            elif transform_type == "kernel_centerer":
                for name in includes:
                    transform_thread.transform_kernel_centerer_arr(df, method_args, name)
            elif transform_type == "yeo_johnson":
                for name in includes:
                    transform_thread.transform_yeo_johnson_arr(df, method_args, name)
            elif transform_type == "box_cox":
                for name in includes:
                    transform_thread.transform_box_cox_arr(df, method_args, name)
            else:
                raise Exception(transform_type)
        except ValueError as exception:
            self.__logger.error(__name__ + " - Invalid configuration(s): " + str(exception))
            sys.exit()

        return df, method_args

    def __transform_df_threaded(self,
                                df: PandasDataFrame,
                                includes: List,
                                transform_type: str,
                                method_args: Dict=None,
                                **kwargs: Any) -> [PandasDataFrame, Dict]:
        """Transform the included features, using the selected and configured method (multi-threaded).
        :param df: the features dataframe.
        :param includes: the name of included features.
        :param transform_type: the transformation arguments, which needs to preserved if it is applied to more than
        one data set.
        :param method_args: the transformation arguments, which needs to preserved if it is applied to more than
        one data set.
        :param kwargs: the input argument for the selected transformation function.
        :return: the transformed feature.
        """
        self.__logger.debug("Transform features (multi-threaded).")
        manager = mp.Manager()
        dt = manager.dict(list(zip(df[includes].columns, df[includes].T.values.tolist())))
        transform_thread = TransformThread(**kwargs)
        method_args = dict() if method_args is None else method_args

        # run
        try:
            with mp.Pool(processes=(mp.cpu_count() - 1)) as pool:
                if transform_type == "scale":
                    pool.map(partial(transform_thread.transform_scale_arr, dt, method_args), includes)
                elif transform_type == "robust_scale":
                    pool.map(partial(transform_thread.transform_robust_scale_arr, dt, method_args), includes)
                elif transform_type == "max_abs_scalar":
                    pool.map(partial(transform_thread.transform_max_abs_scalar_arr, dt, method_args), includes)
                elif transform_type == "normalizer":
                    pool.map(partial(transform_thread.transform_normalizer_arr, dt, method_args), includes)
                elif transform_type == "kernel_centerer":
                    pool.map(partial(transform_thread.transform_kernel_centerer_arr, dt, method_args), includes)
                elif transform_type == "yeo_johnson":
                    pool.map(partial(transform_thread.transform_yeo_johnson_arr, dt, method_args), includes)
                elif transform_type == "box_cox":
                    pool.map(partial(transform_thread.transform_box_cox_arr, dt, method_args), includes)
                else:
                    raise Exception(transform_type)
        except ValueError as exception:
            self.__logger.error(__name__ + " - Invalid configuration(s): " + str(exception))
            sys.exit()

        # set
        for k, v in dt.items():
            df[k] = v

        return df, method_args

    def high_linear_correlation_df(self,
                                   df: PandasDataFrame,
                                   excludes: List,
                                   file_name: str,
                                   thresh_corr_cut: float=0.95,
                                   to_search: bool=True) -> [PandasDataFrame, CollectionsOrderedDict]:
        """Find and optionally remove the selected highly linearly correlated features.
        The Pearson correlation coefficient was calculated for all the pair of variables to measure linear dependence
        between them.
        :param df: the features dataframe.
        :param excludes: the name of excluded features.
        :param file_name: the name of the summary output file.
        :param thresh_corr_cut: the numeric value for the pair-wise absolute correlation cutoff. e.g. 0.95.
        :param to_search: to search or use the saved configuration.
        :return: the inputted dataframe with exclusion of features that were selected to be removed.
        """
        self.__logger.debug("Remove features with high linear correlation (if applicable).")
        corr = None
        df_excludes = df[excludes]
        excludes = set(excludes)
        matches = []
        summaries = OrderedDict()

        # search
        if to_search is True:
            corr = df[[col for col in df.columns if col not in excludes]].corr(method='pearson')
            for label in corr.columns.values:
                matches_temp = list(corr[abs(corr[label]) >= thresh_corr_cut].index)
                if len(matches_temp) > 1:
                    # set matches
                    try:
                        matches_temp.remove(label)
                    except ValueError and AttributeError:
                        pass  # not in some-list! OR not behaving like a list!
                    matches = np.union1d(matches, matches_temp)

                    # summaries
                    for match in matches_temp:
                        if match in summaries.keys():
                            matches_temp.remove(match)
                    if len(matches_temp) > 0:
                        summaries[label] = matches_temp
                        self.__logger.info("High Linear Correlation: " + label + " ~ " + str(matches_temp))

        # delete
        df = self.__remove(df, summaries, to_search, os.path.join(self.__output_path, file_name + ".ini"))
        for name in excludes:
            df[name] = df_excludes[name]
        if any(np.isnan(df.index)):
            df = df.reset_index(drop=True)

        # summaries
        if to_search is True:
            summaries["Features Matches"] = matches
            summaries["Correlation Matrix"] = corr
        return df, summaries

    def near_zero_var_df_sklearn(self,
                                 df: PandasDataFrame,
                                 excludes: List,
                                 file_name: str,
                                 thresh_variance: float=0.05,
                                 to_search: bool=True) -> [PandasDataFrame, CollectionsOrderedDict]:
        """Find and optionally remove the selected near-zero-variance features (Scikit algorithm).
        Feature selector that removes all low-variance features.
        This feature selection algorithm looks only at the features (X), not the desired outputs (y), and can thus be
        used for unsupervised learning.
        :param df: the features dataframe.
        :param excludes: the name of excluded features.
        :param file_name: the name of the summary output file.
        :param thresh_variance: Features with a training-set variance lower than this threshold will be removed.
        The default is to keep all features with non-zero variance, i.e. remove the features that have the same
        value in all samples.
        :param to_search: to search or use the saved configuration.
        :return: the inputted dataframe with exclusion of features that were selected to be removed.
        """
        self.__logger.debug("Remove features with near-zero-variance (if applicable), using Scikit algorithm.")
        df_excludes = df[excludes]
        excludes = set(excludes)
        matches = []
        indices = OrderedDict()
        summaries = OrderedDict()

        # find indices
        for label in df.columns.values():
            indices[df.columns.get_loc(label)] = label

        # search
        if to_search is True:
            variances_ = feature_selection.VarianceThreshold(thresh_variance)
            matches_indices = variances_.get_support(indices=True)
            matches_labels = [indices[index] for index in matches_indices]
            for match in matches_labels:
                if match not in excludes:
                    matches += [match]

        # delete
        df = self.__remove(df, {'NZV': list(matches)}, to_search, os.path.join(self.__output_path, file_name + ".ini"))
        for name in excludes:
            df[name] = df_excludes[name]
        if any(np.isnan(df.index)):
            df = df.reset_index(drop=True)

        # summaries
        if to_search is True:
            summaries["Features Matches"] = matches
        return df, summaries

    def near_zero_var_df(self,
                         df: PandasDataFrame,
                         excludes: List,
                         file_name: str,
                         thresh_unique_cut: float=100,
                         thresh_freq_cut: float=1000,
                         to_search: bool=True) -> [PandasDataFrame, CollectionsOrderedDict]:
        """Find and optionally remove the selected near-zero-variance features (custom algorithm).
        The features that had constant counts less than or equal a threshold may be filtered out,
        to exclude highly constants and near-zero variances.
        Rules are as the following:
         - Frequency ratio: The frequency of the most prevalent value over the second most frequent value to be
           greater than a threshold;
         - Percent of unique values: The number of unique values divided by the total number of samples to be greater
           than the threshold.
        :param df: the features dataframe.
        :param excludes: the name of excluded features.
        :param file_name: the name of the summary output file.
        :param thresh_unique_cut: the cutoff for the percentage of distinct values out of the number of total samples
        (upper limit). e.g. 10 * 100 / 100.
        :param thresh_freq_cut: the cutoff for the ratio of the most common value to the second most common value
        (lower limit). e.g. 95/5.
        :param to_search: to search or use the saved configuration.
        :return: the inputted dataframe with exclusion of features that were selected to be removed.
        """
        self.__logger.debug("Remove features with near-zero-variance (if applicable), using custom algorithm.")
        df_excludes = df[excludes]
        excludes = set(excludes)
        matches = []
        summaries = OrderedDict()

        # search
        if to_search is True:
            for label in df.columns.values:
                # set match and summaries
                # check of NaN
                if not isinstance(df[label].iloc[0], (int, np.int, float, np.float)) \
                        or np.isnan(np.sum(df[label])):
                    matches += [label]
                    continue
                # check of near zero variance
                match, summaries[label] = self.__near_zero_var(
                    df[label], label, excludes, thresh_unique_cut, thresh_freq_cut)
                if match is True:
                    matches += [label]
                    self.__logger.info("Near Zero Variance: " + label)

        # to_remove
        df = self.__remove(df, {'NZV': list(matches)}, to_search, os.path.join(self.__output_path, file_name + ".ini"))
        for name in excludes:
            df[name] = df_excludes[name]
        if any(np.isnan(df.index)):
            df = df.reset_index(drop=True)

        # summaries
        if to_search is True:
            summaries["Features Matches"] = matches
        return df, summaries

    def __near_zero_var(self,
                        arr: List,
                        label: str,
                        excludes: set,
                        thresh_unique_cut: float,
                        thresh_freq_cut: float) -> [bool, Dict]:
        """Assess a single feature for near-zero-variance (custom algorithm).
        The features that had constant counts less than or equal a threshold may be filtered out,
        to exclude highly constants and near-zero variances.
        Rules are as the following:
         - Frequency ratio: The frequency of the most prevalent value over the second most frequent value to be
           greater than a threshold;
         - Percent of unique values: The number of unique values divided by the total number of samples to be greater
           than the threshold.

        :param arr: the feature value.
        :param label: the feature name.
        :param excludes: the name of excluded features.
        :param thresh_unique_cut: the cutoff for the percentage of distinct values out of the number of total samples
        (upper limit). e.g. 10 * 100 / 100.
        :param thresh_freq_cut: the cutoff for the ratio of the most common value to the second most common value
        (lower limit). e.g. 95/5.
        :return: indicates if the feature has near-zero-variance.
        """
        self.__logger.debug("Find near-zero-variance (if applicable), using custom algorithm.")
        unique, counts = np.unique(arr, return_counts=True)
        if len(counts) == 1:
            return True, {'unique': list(unique), 'counts': list(counts)}
        else:
            counts = sorted(counts, reverse=True)
            if label not in excludes and (len(unique) * 100) / float(len(arr)) > thresh_unique_cut:
                return True, {'unique': list(unique), 'counts': list(counts)}
            if label not in excludes and counts[0] / float(counts[1]) > thresh_freq_cut:
                return True, {'unique': list(unique), 'counts': list(counts)}
            else:
                return False, {'unique': list(unique), 'counts': list(counts)}

    def __remove(self,
                 df: PandasDataFrame,
                 dict_matches: Dict,
                 to_search: bool,
                 path: str,
                 section: str="features") -> PandasDataFrame:
        """Confirm removals and if confirmed, then re-read the selected features, then remove
        :param df: the features dataframe.
        :param dict_matches: the matched features.
        :param to_search: to search or use the saved configuration.
        :param path: the file path to the configuration file.
        :param section: the section name in the configuration file.
        :return: the updated features.
        """
        self.__logger.debug("Confirm removals and implement removal process.")
        config = PyConfigParser(path, CONSTANTS.app_name)

        if to_search is True:
            # write to config
            config.reset()
            config.write_dict(dict_matches, section)
            # confirm
            response = self.__readers_writers.question_overwrite(
                "the features defined in the following file to be removed: " + path)
            if response is False:
                config.reset()
                return df

        # if to_search is False or response was yes then read from config
        config.refresh()
        dict_matches = config.read_dict(section)

        # remove
        self.__logger.debug("The feature removal list: " + ",".join(dict_matches))
        labels = [label for label_group in dict_matches.values() for label in label_group if label in df]
        if len(labels) > 0:
            df = df.drop(labels, axis=1)
        return df
