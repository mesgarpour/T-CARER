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
"""It is an interface for connecting to MySQL server.

Help: Installation of MySQL client library:
    - sol. 1 - All OS: pip install MySQL-python
    - sol. 2 - All OS: pip install mysqlclient
    - sol. 3 - Win OS: pip install mysqlclient-***.whl; source: [www.lfd.uci.edu]
"""

from typing import TypeVar
from Configs.CONSTANTS import CONSTANTS
from sqlalchemy import *
from sqlalchemy.pool import NullPool
import sys
import logging

SqlalchemyEngine = TypeVar('Engine')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class MysqlConnection:
    def __init__(self):
        """Initialise the objects and constants."""
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__echo = None
        self.__pool_timeout = None
        self.__pool_recycle = None
        self.__connection_info = None
        self.__status = None
        self.__connection = None
        self.db_session_vars = None

    def set(self,
            db_schema: str):
        """Set the MySQL server configuration settings.
        :param db_schema: the MySQL database schema.
        """
        self.__logger.debug("Set Connection.")
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

    def open(self) -> SqlalchemyEngine:
        """Open a connection to the MySQL server.
        :return: the SQLAlchemy Engine. The Engine is the starting point for any SQLAlchemy application.
        """
        self.__logger.debug("Open Connection.")
        try:
            self.__connection = create_engine(self.__connection_info,
                                              echo=self.__echo,
                                              pool_timeout=self.__pool_timeout,
                                              pool_recycle=self.__pool_recycle)
        except Exception as e:
            self.__logger.error(__name__ + " - DB related error: \n{0:s}.".format(str(e.args[0])))
            sys.exit()

        self.__status = "Open"
        return self.__connection

    def close(self):
        """Close connection to the MySQL server."""
        self.__logger.debug("Close Connection.")
        self.__connection.dispose()
        self.__status = "Close"

    def close_pool(self):
        """Close connection pool to the MySQL server."""
        self.__logger.debug("Close Connection pool.")
        create_engine(self.__connection_info, poolclass=NullPool)
        self.__status = "Close"

    def status(self) -> str:
        """Get the status of connection to the MySQL server.
        :return: the status of connection.
        """
        return self.__status
