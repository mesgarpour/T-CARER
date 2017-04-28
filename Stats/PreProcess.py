#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from collections import OrderedDict
from sklearn import feature_selection
import numpy as np
import pandas as pd
from scipy.stats import stats
import multiprocessing as mp
from functools import partial
import os
import sys
import logging
from Configs.CONSTANTS import CONSTANTS
from ReadersWrites.PyConfigParser import PyConfigParser
from ReadersWrites.ReadersWriters import ReadersWriters
from Stats.FactoringThread import FactoringThread
from Stats.TransformThread import TransformThread

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class PreProcess:
    def __init__(self, output_path):
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__output_path = output_path
        self.__readers_writers = ReadersWriters()

    def stats_odds_ratio(self, df, includes, df_target, target, target_cutoff=0):
        self.__logger.debug(__name__)
        summaries = None
        label_summaries = None
        indices = df_target[df_target[target] > target_cutoff].index.tolist()

        for f_name in includes:
            freq = stats.itemfreq(df[f_name])
            label_freq = stats.itemfreq(df[f_name].loc[indices])

            freq = self.__stats_odds_ratio(f_name, freq, len(df.index))
            label_freq = self.__stats_odds_ratio(f_name, label_freq, len(df.index))

            summaries = summaries.append(freq) if summaries is not None else freq
            label_summaries = summaries.append(label_freq) if label_summaries is not None else label_freq

        summaries = summaries.reset_index()
        label_summaries = label_summaries.reset_index()
        return summaries, label_summaries

    def __stats_odds_ratio(self, f_name, freq, length):
        freq = [[f_name, row[0], row[1], row[1] / length, row[1] / length * 100] for row in freq]
        freq = pd.DataFrame(freq, columns=["Feature_Name", "State", "Odds", "Odds_Ratio", "Odds_Perc"])
        freq["State"] = freq["State"].astype("i4")
        freq["Odds"] = freq["Odds"].astype("i8")
        return freq

    def stats_odds_ratio_conditional(self, df, includes, df_target, target, target_cutoff=1, cond_cutoff=1):
        self.__logger.debug(__name__)
        summaries = None
        label_summaries = None

        for f_name in includes:
            df.ix[df[f_name] >= cond_cutoff, f_name] = 1

            freq = len(df[f_name][df[f_name] == 1])
            label_freq = len(df[(df_target[target] >= target_cutoff) & (df[f_name] == 1)][f_name])

            freq = self.__stats_odds_ratio_conditional(f_name, freq, len(df.index))
            label_freq = self.__stats_odds_ratio_conditional(f_name, label_freq, len(df.index))

            summaries = summaries.append(freq) if summaries is not None else freq
            label_summaries = label_summaries.append(label_freq) if label_summaries is not None else label_freq

        summaries = summaries.reset_index()
        label_summaries = label_summaries.reset_index()
        return summaries, label_summaries

    def __stats_odds_ratio_conditional(self, f_name, freq, length):
        odds_ratio = freq / length if length > 0 else 0
        freq = [[f_name, freq, odds_ratio, odds_ratio * 100]]
        freq = pd.DataFrame(freq, columns=["Feature_Name", "Odds", "Odds_Ratio", "Odds_Perc"])
        freq["Odds"] = freq["Odds"].astype("i8")
        return freq

    def stats_discrete_df(self, df, includes, output_path, file_name):
        self.__logger.debug(__name__)
        summaries = None
        self.__readers_writers.save_csv(path=output_path,
                                        title=file_name,
                                        data=[],
                                        append=False)

        for f_name in df:
            if f_name in includes:
                self.__readers_writers.save_csv(path=output_path,
                                                title=file_name,
                                                data=["Feature Name", f_name],
                                                append=True)
                summaries = stats.itemfreq(df[f_name])
                summaries = pd.DataFrame({"value": summaries[:, 0], "freq": summaries[:, 1]})
                summaries = summaries.sort_values("freq", ascending=False)
                self.__readers_writers.save_csv(path=output_path,
                                                title=file_name,
                                                data=summaries,
                                                append=True,
                                                header=True)
        return summaries

    def stats_continuous_df(self, df, includes, output_path, file_name):
        self.__logger.debug(__name__)
        summaries = None
        self.__readers_writers.save_csv(path=output_path,
                                        title=file_name,
                                        data=[],
                                        append=False)
        for f_name in df:
            if f_name in includes:
                self.__readers_writers.save_csv(path=output_path,
                                                title=file_name,
                                                data=["Feature Name", f_name],
                                                append=True)
                summaries = df[f_name].describe(percentiles=[.25, .5, .75]).transpose()
                summaries = pd.Series.to_frame(summaries).transpose()
                self.__readers_writers.save_csv(path=output_path,
                                                title=file_name,
                                                data=summaries,
                                                append=True,
                                                header=True)
        return summaries

    def factoring_group_wise(self, df, categories_dic, labels_dic, dtypes_dic, threaded=False):
        self.__logger.debug(__name__)
        self.__logger.info("Encoding")
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

    def __factoring_group_wise_series(self, df, categories_dic, labels_dic):
        factoring_thread = FactoringThread(df, categories_dic, labels_dic)
        pool_df_encoded = []

        try:
            for label_group in categories_dic.keys():
                pool_df_encoded.append(factoring_thread.factor_arr_group(label_group))
        except ValueError as exception:
            self.__logger.error(__name__ + " - Invalid configuration(s): " + str(exception))
            sys.exit()
        return pool_df_encoded

    def __factoring_group_wise_threaded(self, df, categories_dic, labels_dic):
        factoring_thread = FactoringThread(df, categories_dic, labels_dic)
        try:
            with mp.Pool(processes=(mp.cpu_count() - 1)) as pool:
                pool_df_encoded = pool.map(
                    partial(factoring_thread.factor_arr_group), categories_dic.keys())
        except ValueError as exception:
            self.__logger.error(__name__ + " - Invalid configuration(s): " + str(exception))
            sys.exit()
        return pool_df_encoded

    def factoring_feature_wise(self, df, categories_dic, labels_dic, dtypes_dic, threaded=False):
        self.__logger.debug(__name__)
        self.__logger.info("Encoding")
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

    def __factoring_feature_wise_series(self, df, categories_dic, labels_dic):
        factoring_thread = FactoringThread(df, categories_dic, labels_dic)
        pool_df_encoded = []

        try:
            for label_group in categories_dic.keys():
                pool_df_encoded.append(factoring_thread.factor_arr(label_group))
        except ValueError as exception:
            self.__logger.error(__name__ + " - Invalid configuration(s): " + str(exception))
            sys.exit()
        return pool_df_encoded

    def __factoring_feature_wise_threaded(self, df, categories_dic, labels_dic):
        factoring_thread = FactoringThread(df, categories_dic, labels_dic)
        try:
            with mp.Pool() as pool:
                pool_df_encoded = pool.map(
                    partial(factoring_thread.factor_arr), categories_dic.keys())
        except ValueError as exception:
            self.__logger.error(__name__ + " - Invalid configuration(s): " + str(exception))
            sys.exit()
        return pool_df_encoded

    def high_linear_correlation_df(self, df, excludes, file_name, thresh_corr_cut=0.95, to_search=True):
        self.__logger.debug(__name__)
        self.__logger.info("Finding high linear correlation (if applicable)")
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
                        print("High Linear Correlation: " + label + " ~ " + str(matches_temp))

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

    def near_zero_var_df_sklearn(self, df, excludes, file_name, thresh_variance=0.05, to_search=True):
        self.__logger.debug(__name__)
        self.__logger.info("Finding near zero variance (if applicable)")
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

    def near_zero_var_df(self, df, excludes, file_name, thresh_unique_cut=100, thresh_freq_cut=1000, to_search=True):
        self.__logger.debug(__name__)
        self.__logger.info("Finding near zero variance (if applicable)")
        df_excludes = df[excludes]
        excludes = set(excludes)
        matches = []
        summaries = OrderedDict()

        # search
        if to_search is True:
            for label in df.columns.values:
                # set match and summaries
                # check of NaN
                if isinstance(df[label][0], str) or np.isnan(np.sum(df[label])):
                    matches += [label]
                    continue
                # check of near zero variance
                match, summaries[label] = self.near_zero_var(
                    df[label], label, excludes, thresh_unique_cut, thresh_freq_cut)
                if match is True:
                    matches += [label]
                    print("Near Zero Variance: " + label)

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

    def near_zero_var(self, arr, label, excludes, thresh_unique_cut, thresh_freq_cut):
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

    def transform_df(self, df, excludes, transform_type, threaded=False, method_args=dict(), **kwargs):
        self.__logger.debug(__name__)
        self.__logger.info("Running transform")
        excludes = set(excludes)
        includes = [label for label in df.columns.values if label not in excludes]

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

    def __transform_df_series(self, df, includes, transform_type, method_args=dict(), **kwargs):
        transform_thread = TransformThread(**kwargs)

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

    def __transform_df_threaded(self, df, includes, transform_type, method_args=dict(), **kwargs):
        manager = mp.Manager()
        dt = manager.dict(list(zip(df[includes].columns, df[includes].T.values.tolist())))
        transform_thread = TransformThread(**kwargs)

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

    def __remove(self, df, dict_matches, to_search, path, section="features"):
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
        self.__logger.debug("Removing: " + ",".join(dict_matches))
        labels = [label for label_group in dict_matches.values() for label in label_group if label in df]
        if len(labels) > 0:
            df = df.drop(labels, axis=1)
        return df
