#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import sys
import logging
from ReadersWrites.PyConfigParser import PyConfigParser

__author__ = 'Mohsen Mesgarpour'
__copyright__ = 'Copyright 2016, https://github.com/mesgarpour'
__credits__ = ['Mohsen Mesgarpour']
__license__ = 'GPL'
__version__ = '1.x'
__maintainer__ = 'Mohsen Mesgarpour'
__email__ = 'mohsen.mesgarpour@gmail.com'
__status__ = 'Development'


class CONSTANTS:
    # configuration settings
    __logger = None
    config_path = None
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

    # Outputs
    output_path = None

    # Input files
    input_path = None
    input_features_configs = None

    # Variables
    var_hesIp = dict()

    @staticmethod
    def set(config_path, app_name, output_path=None):
        CONSTANTS.__logger = logging.getLogger(CONSTANTS.app_name)
        CONSTANTS.__logger.debug(__name__)

        CONSTANTS.config_path = config_path
        CONSTANTS.app_name = app_name

        CONSTANTS.__set_configs_general()

        if output_path is not None:
            CONSTANTS.output_path = output_path
        CONSTANTS.__set_config_other_settings()

    @staticmethod
    def __set_configs_general():
        # configuration settings
        config = PyConfigParser(CONSTANTS.config_path, CONSTANTS.app_name)
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

            # Outputs
            section = 'Outputs'
            CONSTANTS.output_path = str(config.option(section, 'output_path'))

            # Input files
            section = 'Inputs'
            CONSTANTS.input_path = str(config.option(section, 'input_path'))
            CONSTANTS.input_features_configs = str(config.option(section, 'input_features_configs'))
        except():
            CONSTANTS.__logger.error(__name__ + " - Invalid configuration(s) in the " + section + " section")
            sys.exit()

    @staticmethod
    def __set_config_other_settings():
        CONSTANTS.__create_directories(CONSTANTS.output_path)
        CONSTANTS.output_path = os.path.abspath(CONSTANTS.output_path)
        CONSTANTS.input_path = os.path.abspath(CONSTANTS.input_path)

    @staticmethod
    def __create_directories(path):
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as e:
                CONSTANTS.__logger.error(
                    __name__ + ' - Directory creation error: {0:d}:\n {1:s}'.format(e.args[0], str(e.args[1])))
                sys.exit()
