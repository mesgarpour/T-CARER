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
""" It configures the Python application logger.
"""

import os
import logging

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class Logger:
    def __init__(self,
                 path: str,
                 app_name: str,
                 ext: str = "log"):
        """Initialise the objects and constants.
        :param path: the output directory path, where the log file will be saved.
        :param app_name: the application name, which will be used as the log file name.
        :param ext: the log file extension name.
        """
        # create logger
        logger = logging.getLogger(app_name)
        logger.setLevel(logging.DEBUG)
        path_full = os.path.abspath(os.path.join(path, app_name + "." + ext))

        # create file handler which logs even debug messages
        fh = logging.FileHandler(path_full, mode='w')
        fh.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create formatter and add it to the handlers
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        # output log
        logger.info("Creating '" + path_full + "' File.")
