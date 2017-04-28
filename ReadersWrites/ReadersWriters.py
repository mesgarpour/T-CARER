#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from ReadersWrites._TextFile import TextFile
from ReadersWrites._CsvFile import _CsvFile
from ReadersWrites._MysqlCommand import MysqlCommand
from ReadersWrites._MysqlConnection import MysqlConnection
from ReadersWrites._MysqlQueries import MysqlQueries
from ReadersWrites._PickleSerialised import PickleSerialised
import math
import os
import sys

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class ReadersWriters:

    @staticmethod
    def exists_serialised(path, title, ext):
        reader = PickleSerialised()
        reader.set(path, title, ext)
        return reader.exists()

    @staticmethod
    def exists_mysql(schema, table):
        queries = MysqlQueries()
        query = queries.exists_table(table)
        return len(ReadersWriters.load_mysql_query(query, schema, dataframing=True)) > 0

    @staticmethod
    def exists_csv(path, title, extension="csv"):
        reader = _CsvFile()
        reader.set(path, title, extension)
        return reader.exists()

    @staticmethod
    def exists_text(path, title, extension):
        reader = TextFile()
        reader.set(path, title, extension)
        return reader.exists()

    @staticmethod
    def exists_mysql_column(schema, table, column):
        queries = MysqlQueries()
        query = queries.exists_column(table, column)
        return len(ReadersWriters.load_mysql_query(query, schema, dataframing=True)) > 0

    @staticmethod
    def exists_csv_column(path, title, column, extension):
        reader = _CsvFile()
        reader.set(path, title, extension)
        return reader.exists_column(column)

    @staticmethod
    def size_serialised(path, title, ext):
        reader = PickleSerialised()
        reader.set(path, title, ext)
        return reader.size()

    @staticmethod
    def size_mysql_table(schema, table):
        return int(ReadersWriters.load_mysql_table(schema, table, dataframing=True)[0][0])

    @staticmethod
    def size_mysql_query(query, schema):
        return int(ReadersWriters.load_mysql_query(query, schema, dataframing=True)[0][0])

    @staticmethod
    def size_csv(path, title, extension="csv"):
        reader = _CsvFile()
        reader.set(path, title, extension)
        return reader.size()

    @staticmethod
    def size_text(path, title, extension):
        reader = TextFile()
        reader.set(path, title, extension)
        return reader.size()

    @staticmethod
    def save_serialised(path, title, objects):
        writer = PickleSerialised()
        writer.set(path, title, "pickle")
        writer.save(objects)

    @staticmethod
    def save_serialised_compressed(path, title, objects):
        writer = PickleSerialised()
        writer.set(path, title, "bz2")
        writer.save_bz2(objects)

    @staticmethod
    def save_mysql(query, data, schema, table, batch=None):
        db = MysqlConnection()
        db.set(schema)
        engine = db.open()
        dbc = MysqlCommand(engine, db.db_session_vars)
        if data is None:
            dbc.write(query)
        else:
            if batch is None:
                dbc.write_many(query, data, schema, table)
            else:
                size = math.ceil(data.shape[0] / batch)
                for i in range(0, size):
                    dbc.write_many(query, data[i * batch:(i + 1) * batch], schema, table)

        db.close()

    @staticmethod
    def save_csv(path, title, data, append=False, extension="csv", **kwargs):
        writer = _CsvFile()
        writer.set(path, title, extension)
        if append is False:
            writer.reset()
        writer.append(data, **kwargs)

    @staticmethod
    def save_text(path, title, data, append=False, extension="log"):
        writer = TextFile()
        writer.set(path, title, extension)
        if append is False:
            writer.reset()
        writer.append(data)

    @staticmethod
    def load_serialised(path, title):
        reader = PickleSerialised()
        reader.set(path, title, "pickle")
        return reader.load()

    @staticmethod
    def load_serialised_compressed(path, title):
        reader = PickleSerialised()
        reader.set(path, title, "bz2")
        return reader.load_bz2()

    @staticmethod
    def load_mysql_table(schema, table, dataframing=True):
        query = "SELECT * FROM " + table
        return ReadersWriters.load_mysql_query(query, schema, dataframing)

    @staticmethod
    def load_mysql_query(query, schema, dataframing=True, batch=None, float_round_vars=None, float_round=None):
        db = MysqlConnection()
        db.set(schema)
        engine = db.open()
        dbc = MysqlCommand(engine, db.db_session_vars)
        output = dbc.read(query, dataframing, batch, float_round_vars, float_round)
        db.close()
        return output

    @staticmethod
    def load_csv(path, title, skip=0, dataframing=True, extension="csv", **kwargs):
        reader = _CsvFile()
        reader.set(path, title, extension)
        return reader.read(skip, dataframing, **kwargs)

    @staticmethod
    def load_text(path, title, extension, skip):
        reader = TextFile()
        reader.set(path, title, extension)
        reader.read(skip)

    @staticmethod
    def load_mysql_procedure(query, args, schema):
        db = MysqlConnection()
        db.set(schema)
        engine = db.open()
        dbc = MysqlCommand(engine, db.db_session_vars)
        output = dbc.call_proc(query, args)
        db.close()
        return output

    @staticmethod
    def reset_mysql_table(schema, table):
        query = "TRUNCATE TABLE " + table
        return ReadersWriters.load_mysql_query(query, schema)

    @staticmethod
    def reset_csv(path, title, extension="csv"):
        reader = _CsvFile()
        reader.set(path, title, extension)
        reader.reset()

    @staticmethod
    def reset_text(path, title, extension="log"):
        reader = TextFile()
        reader.set(path, title, extension)
        reader.reset()

    @staticmethod
    def mysql_query_create(table, columns, types, defaults, primary_keys=list(), unique_keys=list(),
                           meta="ENGINE=InnoDB DEFAULT CHARSET=latin1"):
        queries = MysqlQueries()
        return queries.create(table, columns, types, defaults, primary_keys, unique_keys, meta)

    @staticmethod
    def mysql_query_insert(table, columns, types):
        queries = MysqlQueries()
        return queries.insert(table, columns, types)

    @staticmethod
    def mysql_query_drop(table):
        queries = MysqlQueries()
        return queries.drop(table)

    @staticmethod
    def question_overwrite(name):
        while True:
            response = input("Confirm or reject " + name + "\n >> Print \'y\' to accept or \'n\' to decline: ").lower()
            if response == 'y':
                print("Approved")
                return True
            elif response == 'n':
                print("Declined")
                return False
            else:
                print("ERROR: Invalid command \'y\' or \'n\'")

    @staticmethod
    def create_directories(path):
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as e:
                sys.exit(__name__ + ' - Directory creation error: {0:d}:\n {1:s}'.format(e.args[0], str(e.args[1])))
