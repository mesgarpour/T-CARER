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
""" It computes the Yeo-Johnson transofrmation, which is an extension of Box-Cox transformation
but can handle both positive and negative values.

References:
Weisberg, S. (2001). Yeo-Johnson Power Transformations.
Department of Applied Statistics, University of Minnesota. Retrieved June, 1, 2003.
https://www.stat.umn.edu/arc/yjpower.pdf

Adapted from CRAN - Package VGAM
"""

from typing import List, TypeVar, Callable
import sys
import warnings
import numpy as np
import pandas as pd

NumpyNDArray = TypeVar('ndarray')
PandasSeries = TypeVar('Series')


class YeoJohnson:
    def fit(self,
            y: Callable[[List, NumpyNDArray, PandasSeries], None],
            lmbda: Callable[[int, float], None],
            derivative: Callable[[int, float], None]=0,
            epsilon: Callable[[int, float], None]=np.finfo(np.float).eps,
            inverse: bool=False):
        """Calculate the yeo-johnson transformation for a feature.
        :param y: the variable to be transformed (numeric array).
        :param lmbda: the function's Lambda value (numeric value or array).
        :param derivative: the derivative with respect to lambda.
        (non-negative integer; default: ordinary function evaluation).
        :param epsilon: the lambda's tolerance (positive value).
        :param inverse: the inverse transformation option (logical value).
        :return: the Yeo-Johnson transformation or its inverse, or its derivatives with respect to lambda, of y.
        """
        # Validate arguments
        self.__validate(y, lmbda, derivative, epsilon, inverse)

        # initialise
        y = np.array(y, dtype=float)
        result = y
        if not (isinstance(lmbda, list) or isinstance(lmbda, np.ndarray)):
            lmbda, y = np.broadcast_arrays(lmbda, y)
            lmbda = np.array(lmbda, dtype=float)
        l0 = np.abs(lmbda) > epsilon
        l2 = np.abs(lmbda - 2) > epsilon

        # inverse
        with warnings.catch_warnings():  # suppress warnings
            warnings.simplefilter("ignore")
            if inverse is True:
                mask = np.where(((y >= 0) & l0) is True)
                result[mask] = np.power(np.multiply(y[mask], lmbda[mask]) + 1, 1 / lmbda[mask]) - 1

                mask = np.where(((y >= 0) & ~l0) is True)
                result[mask] = np.expm1(y[mask])

                mask = np.where(((y < 0) & l2) is True)
                result[mask] = 1 - np.power(np.multiply(-(2 - lmbda[mask]), y[mask]) + 1, 1 / (2 - lmbda[mask]))

                mask = np.where(((y < 0) & ~l2) is True)
                result[mask] = -np.expm1(-y[mask])

            # derivative
            else:
                if derivative == 0:
                    mask = np.where(((y >= 0) & l0) is True)
                    result[mask] = np.divide(np.power(y[mask] + 1, lmbda[mask]) - 1, lmbda[mask])

                    mask = np.where(((y >= 0) & ~l0) is True)
                    result[mask] = np.log1p(y[mask])

                    mask = np.where(((y < 0) & l2) is True)
                    result[mask] = np.divide(-(np.power(-y[mask] + 1, 2 - lmbda[mask]) - 1), 2 - lmbda[mask])

                    mask = np.where(((y < 0) & ~l2) is True)
                    result[mask] = -np.log1p(-y[mask])

                # Not derivative
                else:
                    p = self.fit(y, lmbda, derivative=derivative - 1, epsilon=epsilon, inverse=inverse)

                    mask = np.where(((y >= 0) & l0) is True)
                    result[mask] = np.divide(np.multiply(
                        np.power(y[mask] + 1,
                                 lmbda[mask]),
                        np.power(np.log1p(y[mask]),
                                 derivative)) - np.multiply(derivative, p[mask]), lmbda[mask])

                    mask = np.where(((y >= 0) & ~l0) is True)
                    result[mask] = np.divide(np.power(np.log1p(y[mask]), derivative + 1), derivative + 1)

                    mask = np.where(((y < 0) & l2) is True)
                    result[mask] = np.divide(-(np.multiply(
                        np.power(-y[mask] + 1,
                                 2 - lmbda[mask]),
                        np.power(-np.log1p(-y[mask]),
                                 derivative)) - np.multiply(derivative, p[mask])), 2 - lmbda[mask])

                    mask = np.where(((y < 0) & ~l2) is True)
                    result[mask] = np.divide(np.power(-np.log1p(-y[mask]), derivative + 1), derivative + 1)
        return result

    @staticmethod
    def __validate(y: Callable[[List, NumpyNDArray, PandasSeries], None],
                   lmbda: Callable[[int, float], None],
                   derivative: Callable[[int, float], None],
                   epsilon: Callable[[int, float], None],
                   inverse: bool):
        """Validate the input arguments.
        :param y: the variable to be transformed (numeric array).
        :param lmbda: the function's Lambda value (numeric value or array).
        :param derivative: the derivative with respect to lambda.
        (non-negative integer; default: ordinary function evaluation).
        :param epsilon: the lambda's tolerance (positive value).
        :param inverse: the inverse transformation option (logical value).
        """
        try:
            if not isinstance(y, (list, np.ndarray, pd.Series)):
                raise Exception("Argument 'y' must be a list")
            if not isinstance(lmbda, (int, float, np.int, np.float)):
                if not isinstance(lmbda, (list, np.ndarray, pd.Series)) or len(lmbda) != len(y):
                    raise Exception("Argument 'lmbda' must be a number "
                                    "or a list, which its length matches 'y' argument")
            if not isinstance(derivative, (int, float, np.int, np.float)) or derivative < 0:
                raise Exception("Argument 'derivative' must be a non-negative integer")
            if not isinstance(epsilon, (int, float, np.int, np.float)) or epsilon <= 0:
                raise Exception("Argument 'epsilon' must be a positive number")
            if not isinstance(inverse, bool):
                raise Exception("Argument 'inverse' must be boolean")
            if inverse is True and derivative != 0:
                raise Exception("Argument 'derivative' must be zero "
                                "when argument 'inverse' is 'True'")
        except ():
            sys.exit()
