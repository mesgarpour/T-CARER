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
""" It reads and sets constants, based on the configuration file and input arguments.
"""

import os
import sys
import logging
from ReadersWriters.PyConfigParser import PyConfigParser

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class CONSTANTS:
    # configuration settings
    __logger = None
    app_name = None

    # Database
    db_host = None
    db_user = None
    db_password = None
    db_port = None
    db_echo = None
    db_pool_timeout = None
    db_pool_recycle = None
    db_session_vars = None

    # Inputs and Outputs
    io_path = None
    config_path = None
    config_features_path = None

    @staticmethod
    def set(io_path: str,
            app_name: str="logs",
            config_features_path: str="ConfigInputs/input_features_configs",
            config_path: str="ConfigInputs/CONFIGURATIONS"):
        """Configure configuration file and set the constants.
        :param io_path: the input output directory path.
        :param app_name: the application name, which will be used as the log file name.
        :param config_features_path: the configuration of the input features.
        :param config_path: the configuration directory path.
        """
        CONSTANTS.__logger = logging.getLogger(CONSTANTS.app_name)
        CONSTANTS.__logger.debug(__name__)

        CONSTANTS.io_path = io_path
        CONSTANTS.config_path = os.path.abspath(config_path)
        CONSTANTS.config_features_path = os.path.abspath(config_features_path)
        CONSTANTS.app_name = app_name

        CONSTANTS.__set_configs_general()
        CONSTANTS.__set_config_other_settings()

    @staticmethod
    def __set_configs_general():
        """Set general configuration constants, including the MySQL database constants.
        """
        # configuration settings
        config = PyConfigParser(CONSTANTS.config_path, CONSTANTS.app_name, ext="ini")
        section = ""

        try:
            # Database
            section = 'Database'
            CONSTANTS.db_host = str(config.option(section, 'db_host'))
            CONSTANTS.db_user = str(config.option(section, 'db_user'))
            CONSTANTS.db_password = str(config.option(section, 'db_password'))
            CONSTANTS.db_port = str(config.option(section, 'db_port'))
            CONSTANTS.db_echo = str(config.option(section, 'db_echo')) == 'True'
            CONSTANTS.db_pool_timeout = int(config.option(section, 'db_pool_timeout'))
            CONSTANTS.db_pool_recycle = int(config.option(section, 'db_pool_recycle'))
            CONSTANTS.db_session_vars = str(config.option(section, 'db_session_vars')).split(';')
        except():
            CONSTANTS.__logger.error(__name__ + " - Invalid configuration(s) in the " + section + " section")
            sys.exit()

    @staticmethod
    def __set_config_other_settings():
        """Set other configuration constants.
        """
        CONSTANTS.__create_directories(CONSTANTS.io_path)
        CONSTANTS.io_path = os.path.abspath(CONSTANTS.io_path)

    @staticmethod
    def __create_directories(path: str):
        """Create folder if it does not exist.
        :param path: Directory path.
        """
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as e:
                CONSTANTS.__logger.error(
                    __name__ + ' - Directory creation error: {0:d}:\n {1:s}'.format(e.args[0], str(e.args[1])))
                sys.exit()
