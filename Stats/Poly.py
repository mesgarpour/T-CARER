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
"""It applies polynomial function
Adapted from http://davmre.github.io/
"""
import numpy as np
import sys

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class Poly:
    @staticmethod
    def train(x, degree=1):
        try:
            n = degree + 1
            x = np.asarray(x).flatten()
            if degree >= len(np.unique(x)):
                raise Exception("'degree' must be less than number of unique points")
            xbar = np.mean(x)
            x -= xbar
            X = np.fliplr(np.vander(x, n))
            q, r = np.linalg.qr(X)

            z = np.diag(np.diag(r))
            raw = np.dot(q, z)

            norm2 = np.sum(raw ** 2, axis=0)
            alpha = (np.sum((raw ** 2) * np.reshape(x, (-1, 1)), axis=0) / norm2 + xbar)[:degree]
            Z = raw / np.sqrt(norm2)
        except ():
            sys.exit()

        return Z, norm2, alpha

    @staticmethod
    def predict(x, alpha, norm2, degree=1):
        x = np.asarray(x).flatten()
        n = degree + 1
        Z = np.empty((len(x), n))

        Z[:, 0] = 1
        if degree > 0:
            Z[:, 1] = x - alpha[0]
        if degree > 1:
            for i in np.arange(1, degree):
                Z[:, i+1] = (x - alpha[i]) * Z[:, i] - (norm2[i] / norm2[i-1]) * Z[:, i-1]
        Z /= np.sqrt(norm2)
        return Z
