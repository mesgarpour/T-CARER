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
""" It applies a set of transformation functions using independent threads for each feature.
"""

from typing import TypeVar, Any
from scipy import stats
from sklearn import preprocessing
from Stats.YeoJohnson import YeoJohnson
import numpy as np

PandasDataFrame = TypeVar('DataFrame')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class TransformThread:
    # todo: optimise threading further

    def __init__(self,
                 **kwargs: Any):
        """Initialise the objects and constants.
        :param kwargs: the input arguments for the selected transform function.
        """
        self.__kwargs = kwargs

    def transform_scale_arr(self,
                            dt: PandasDataFrame,
                            method_args: Any,
                            name: str):
        """Standardize a dataset along any axis.
        :param dt: the dataframe of features.
        :param method_args: other input arguments
        (kwargs: with_mean=True)
        :param name: the name of the feature to be transformed.
        """
        method_args[name] = None
        dt[name] = preprocessing.scale(dt[name], **self.__kwargs)

    def transform_robust_scale_arr(self,
                                   dt: PandasDataFrame,
                                   method_args: Any,
                                   name: str):
        """Standardize a dataset along any axis.
        :param dt: the dataframe of features.
        :param method_args: other input arguments
        (kwargs: axis=0, with_centering=True, with_scaling=True)
        :param name: the name of the feature to be transformed.
        """
        method_args[name] = None
        dt[name] = preprocessing.robust_scale(dt[name], **self.__kwargs)

    def transform_max_abs_scalar_arr(self,
                                     dt: PandasDataFrame,
                                     method_args: Any,
                                     name: str):
        """Scale each feature by its maximum absolute value.
        :param dt: the dataframe of features.
        :param method_args: other input arguments
        (it is a placeholder no argument is available).
        :param name: the name of the feature to be transformed.
        """
        if name in method_args[name] and "scale" in method_args[name].keys():
            scale = method_args[name]["scale"]
        else:
            scale = preprocessing.MaxAbsScaler(**self.__kwargs)
            method_args[name] = {"scale": scale}

        arr = scale.fit_transform(dt[name])
        arr = np.array(scale.transform(arr)) + 1
        dt[name], summaries = stats.boxcox(arr)

    def transform_normalizer_arr(self,
                                 dt: PandasDataFrame,
                                 method_args: Any,
                                 name: str):
        """Normalize samples individually to unit norm.
        :param dt: the dataframe of features.
        :param method_args: other input arguments
        (kwargs: norm='l2')
        :param name: the name of the feature to be transformed.
        """
        if name in method_args[name] and "scale" in method_args[name].keys():
            scale = method_args[name]["scale"]
        else:
            scale = preprocessing.Normalizer(**self.__kwargs)
            method_args[name] = {"scale": scale}

        arr = scale.fit_transform(dt[name])
        dt[name] = scale.transform(arr)

    def transform_kernel_centerer_arr(self,
                                      dt: PandasDataFrame,
                                      method_args: Any,
                                      name: str):
        """Center a kernel matrix
        :param dt: the dataframe of features.
        :param method_args: other input arguments
        (it is a placeholder no argument is available).
        :param name: the name of the feature to be transformed.
        """
        if name in method_args[name] and "scale" in method_args[name].keys():
            scale = method_args[name]["scale"]
        else:
            scale = preprocessing.KernelCenterer()
            method_args[name] = {"scale": scale}

        arr = scale.fit_transform(dt[name])
        dt[name] = scale.transform(arr)

    def transform_yeo_johnson_arr(self,
                                  dt: PandasDataFrame,
                                  method_args: Any,
                                  name: str):
        """Apply the Ye-Johnson transformation.
        :param dt: the dataframe of features.
        :param method_args: other input arguments
        (kwargs: lmbda=-0.5, derivative=0, epsilon=np.finfo(np.float).eps, inverse=False).
        :param name: the name of the feature to be transformed.
        """
        method_args[name] = None
        yeo_johnson = YeoJohnson()
        dt[name] = yeo_johnson.fit(dt[name], **self.__kwargs)

    def transform_box_cox_arr(self,
                              dt: PandasDataFrame,
                              method_args: Any,
                              name: str):
        """Apply the Box-Cox transformation.
        :param dt: the dataframe of features.
        :param method_args: other input arguments
        (kwargs: lmbda=None, alpha=None).
        :param name: the name of the feature to be transformed.
        """
        if name in method_args[name] and "scale" in method_args[name].keys():
            scale = method_args[name]["scale"]
        else:
            scale, _ = stats.boxcox(dt[name], **self.__kwargs)
            method_args[name] = {"scale": scale}

        arr = scale.fit_transform(dt[name])
        dt[name] = scale.transform(arr)
