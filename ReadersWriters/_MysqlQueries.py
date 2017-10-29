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
""" Generate custom MySQL queries.
"""

from typing import List
import sys
import logging
from Configs.CONSTANTS import CONSTANTS

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class MysqlQueries:
    def __init__(self):
        """Initialise the objects and constants.
        """
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)

    def create(self,
               db_table: str,
               columns: List,
               col_types: List,
               defaults: List,
               primary_keys: List=list(),
               unique_keys: List=list(),
               meta: str="ENGINE=InnoDB DEFAULT CHARSET=latin1") -> str:
        """Generate a custom 'CREATE TABLE' query.
        :param db_table: the MySQL table.
        :param columns: the list of columns.
        :param col_types: the list of column types for columns.
        :param defaults: the list of default values for columns.
        :param primary_keys: the list of primary keys for the table.
        :param unique_keys: the list of unique keys for the table.
        :param meta: the table metadata (Engine and charset).
        :return: MySQL query.
        """
        self.__logger.debug("Generate custom 'CREATE TABLE' MySQL query.")
        if not(len(columns) == len(col_types) == len(defaults)):
            self.__logger.error(__name__ + " - Can not create the specified table \n" +
                                " Num. Columns: " + str(len(columns)) +
                                " Num. metadata: " + str(len(col_types)))
            sys.exit()

        # name
        query = "CREATE TABLE " + db_table + " ("

        # columns
        for i in range(len(columns)):
            query += columns[i] + " " + col_types[i] + " " + defaults[i] + ","
        query = query[:-1]

        # primary keys
        primary_keys_t = []
        for k in range(len(primary_keys)):
            if primary_keys[k] is True:
                primary_keys_t.append(columns[k])
        if len(primary_keys_t) > 0:
            query += ",PRIMARY KEY("
            for primary_key in primary_keys_t:
                query += primary_key + ","
            query = query[:-1] + ")"

        # unique keys
        unique_keys_t = []
        for k in range(len(unique_keys)):
            if unique_keys[k] is True:
                unique_keys_t.append(columns[k])
        if len(unique_keys_t) > 0:
            for unique_key in unique_keys_t:
                query += ",UNIQUE KEY " + unique_key + "(" + unique_key + ")"

        # meta data
        query += ") " + meta
        self.__logger.debug(query)
        return query

    def insert(self,
               db_table: str,
               columns: List,
               col_types: List) -> str:
        """Generate a custom 'INSERT INTO' query.
        :param db_table: the MySQL table.
        :param columns: the list of columns.
        :param col_types: the list of column types for columns.
        :return: MySQL query.
        """
        self.__logger.debug("Generate custom 'INSERT INTO' MySQL query.")
        # name
        query = "INSERT INTO " + db_table + " ("

        # columns
        for column in columns:
            query += column + ","
        query = query[:-1] + ")"

        # values placeholders
        query += " VALUES"
        for t in col_types:
            t = t.lower()
            if t[0:7] == "varchar" \
                    or t[0:4] == "char" \
                    or t[0:8] == "tinyblob" \
                    or t[0:8] == "tinytext" \
                    or t[0:4] == "blob" \
                    or t[0:4] == "text" \
                    or t[0:10] == "mediumblob" \
                    or t[0:10] == "mediumtext" \
                    or t[0:8] == "longblob" \
                    or t[0:8] == "longtext":
                query += "%s,"
            elif t[0:3] == "int" \
                    or t[0:7] == "integer" \
                    or t[0:7] == "tinyint" \
                    or t[0:8] == "smallint" \
                    or t[0:9] == "mediumint":
                query += "%i,"
            elif t[0:5] == "float" \
                    or t[0:6] == "double" \
                    or t[0:4] == "real" \
                    or t[0:7] == "decimal" \
                    or t[0:7] == "numeric":
                query += "%f,"
            else:
                self.__logger.error(__name__ + " - invalid insert type: \n" + str(t))
                sys.exit()
        query = query[:-1] + ")"

        # values
        query += " ON DUPLICATE KEY UPDATE "
        for column in columns:
            query += column + "=VALUES(" + column + "),"
        query = query[:-1]
        self.__logger.debug(query)
        return query

    def drop(self,
             db_table: str) -> str:
        """Generate a custom 'DROP TABLE IF EXISTS' query.
        :param db_table: the MySQL table.
        :return: MySQL query.
        """
        self.__logger.debug("Generate custom 'DROP TABLE IF EXISTS' MySQL query.")
        query = "DROP TABLE IF EXISTS " + db_table
        self.__logger.debug(query)
        return query

    def exists_table(self,
                     db_table: str) -> str:
        """Generate a custom query to search 'information_schema.tables' for a particular table.
        :param db_table: the MySQL table.
        :return: MySQL query.
        """
        self.__logger.debug("Generate a MySQL query to search for a particular table.")
        query = "SELECT * FROM information_schema.tables " + \
                "WHERE TABLE_NAME = \'" + db_table + "\'"
        self.__logger.debug(query)
        return query

    def exists_column(self,
                      db_table: str,
                      column: str) -> str:
        """Generate a custom query to search 'information_schema.tables' for a particular column in a table.
        :param db_table: the MySQL table.
        :param column: the table's column.
        :return: MySQL query.
        """
        self.__logger.debug("Generate a MySQL query to search for a particular table's column.")
        query = "SELECT * FROM information_schema.tables " + \
                "WHERE TABLE_NAME = \'" + db_table + "\' " + \
                "AND COLUMN_NAME = \'" + column + "\'"
        self.__logger.debug(query)
        return query
