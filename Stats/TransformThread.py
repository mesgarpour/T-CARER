#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from scipy import stats
from sklearn import preprocessing
from Stats.YeoJohnson import YeoJohnson

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class TransformThread:
    def __init__(self, **kwargs):
        self.__kwargs = kwargs

    def transform_scale_arr(self, dt, method_args, name):
        """
        kwargs: 
        with_mean=True
        """
        method_args[name] = None
        dt[name] = preprocessing.scale(dt[name], **self.__kwargs)

    def transform_robust_scale_arr(self, dt, method_args, name):
        """
        kwargs: 
        axis=0, with_centering=True, with_scaling=True
        """
        method_args[name] = None
        dt[name] = preprocessing.robust_scale(dt[name], **self.__kwargs)

    def transform_max_abs_scalar_arr(self, dt, method_args, name):
        if name in method_args[name] and "scale" in method_args[name].keys():
            scale = method_args[name]["scale"]
        else:
            scale = preprocessing.MaxAbsScaler(**self.__kwargs)
            method_args[name] = {"scale": scale}

        temp = scale.fit_transform(dt[name])
        temp = scale.transform(temp)
        dt[name], summaries = stats.boxcox(temp + 1)

    def transform_normalizer_arr(self, dt, method_args, name):
        """
        kwargs: 
        norm='l2'
        """
        if name in method_args[name] and "scale" in method_args[name].keys():
            scale = method_args[name]["scale"]
        else:
            scale = preprocessing.Normalizer(**self.__kwargs)
            method_args[name] = {"scale": scale}

        arr = scale.fit_transform(dt[name])
        dt[name] = scale.transform(arr)

    def transform_kernel_centerer_arr(self, dt, method_args, name):
        if name in method_args[name] and "scale" in method_args[name].keys():
            scale = method_args[name]["scale"]
        else:
            scale = preprocessing.KernelCenterer()
            method_args[name] = {"scale": scale}

        arr = scale.fit_transform(dt[name])
        dt[name] = scale.transform(arr)

    def transform_yeo_johnson_arr(self, dt, method_args, name):
        """
        kwargs: 
        lmbda=-0.5, derivative=0, epsilon=np.finfo(np.float).eps, inverse=False
        """
        method_args[name] = None
        yeo_johnson = YeoJohnson()
        dt[name] = yeo_johnson.fit(dt[name], **self.__kwargs)

    def transform_box_cox_arr(self, dt, method_args, name):
        """
        kwargs: 
        lmbda=None, alpha=None
        """
        if name in method_args[name] and "scale" in method_args[name].keys():
            scale = method_args[name]["scale"]
        else:
            scale, _ = stats.boxcox(dt[name], **self.__kwargs)
            method_args[name] = {"scale": scale}

        arr = scale.fit_transform(dt[name])
        dt[name] = scale.transform(arr)
