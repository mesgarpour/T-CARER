#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from Configs.CONSTANTS import CONSTANTS
from sqlalchemy import *
from sqlalchemy.pool import NullPool
import sys
import logging

# MySQL library dependencies:
# [sol. 1]: All OS: pip install MySQL-python
# [sol. 2]: All OS: pip install mysqlclient
# [sol. 3 - www.lfd.uci.edu]: pip install mysqlclient-....whl

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class MysqlConnection:
    """Handle MySQL connections"""

    def __init__(self):
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__echo = None
        self.__pool_timeout = None
        self.__pool_recycle = None
        self.__connection_info = None
        self.__status = None
        self.__connection = None
        self.db_session_vars = None

    def set(self, db_schema):
        self.__logger.debug(__name__)
        self.__echo = CONSTANTS.db_echo
        self.__pool_timeout = CONSTANTS.db_pool_timeout
        self.__pool_recycle = CONSTANTS.db_pool_recycle
        self.db_session_vars = CONSTANTS.db_session_vars
        self.__connection_info = 'mysql+mysqldb://{}:{}@{}:{}/{}'.format(
            CONSTANTS.db_user,
            CONSTANTS.db_password,
            CONSTANTS.db_host,
            str(CONSTANTS.db_port),
            db_schema)

    def open(self):
        """Open Connection"""
        self.__logger.debug(__name__)
        try:
            self.__connection = create_engine(self.__connection_info,
                                              echo=self.__echo,
                                              pool_timeout=self.__pool_timeout,
                                              pool_recycle=self.__pool_recycle)

        except Exception as e:
            self.__logger.error(__name__ + " - DB related error: {0:s}".format(str(e.args[0])))
            sys.exit()

        self.__status = "Open"
        return self.__connection

    def close(self):
        """Close connection"""
        self.__logger.debug(__name__)
        self.__connection.dispose()
        self.__status = "Close"

    def close_pool(self):
        """Close connection"""
        self.__logger.debug(__name__)
        create_engine(self.__connection_info, poolclass=NullPool)
        self.__status = "Close"

    def status(self):
        """Get Connection status"""
        return self.__status
