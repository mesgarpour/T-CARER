#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import logging
from Configs.CONSTANTS import CONSTANTS

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class MysqlQueries:
    __logger = logging.getLogger(CONSTANTS.app_name)
    __logger.debug(__name__)

    @staticmethod
    def create(table, columns, types, defaults, primary_keys=list(), unique_keys=list(),
               meta="ENGINE=InnoDB DEFAULT CHARSET=latin1"):
        if not(len(columns) == len(types) == len(defaults)):
            MysqlQueries.__logger.error(__name__ + " - Can not create the specified table" +
                                        " Num. Columns: " + str(len(columns)) +
                                        " Num. metadata: " + str(len(types)))
            sys.exit()

        # name
        query = "CREATE TABLE " + table + " ("

        # columns
        for i in range(len(columns)):
            query += columns[i] + " " + types[i] + " " + defaults[i] + ","
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
        return query

    @staticmethod
    def insert(table, columns, types):
        # name
        query = "INSERT INTO " + table + " ("

        # columns
        for column in columns:
            query += column + ","
        query = query[:-1] + ")"

        # values placeholders
        query += " VALUES"
        for t in types:
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
                MysqlQueries.__logger.error(__name__ + " - invalid insert type: " + str(t))
                sys.exit()
        query = query[:-1] + ")"

        # values
        query += " ON DUPLICATE KEY UPDATE "
        for column in columns:
            query += column + "=VALUES(" + column + "),"
        query = query[:-1]
        MysqlQueries.__logger.debug(query)
        return query

    @staticmethod
    def drop(table):
        return "DROP TABLE IF EXISTS " + table

    @staticmethod
    def exists_table(table):
        return "SELECT * FROM information_schema.tables " + \
               "WHERE TABLE_NAME = \'" + table + "\'"

    @staticmethod
    def exists_column(table, column):
        return "SELECT * FROM information_schema.tables " + \
               "WHERE TABLE_NAME = \'" + table + "\' " + \
               "AND COLUMN_NAME =  \'" + column + "\'"
