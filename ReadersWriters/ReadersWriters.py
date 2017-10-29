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
""" It is an interface for the developed readers and writers.
"""

from typing import List, TypeVar, Dict, Any, Callable
import math
import os
import sys
from ReadersWriters._CsvFile import CsvFile
from ReadersWriters._MysqlCommand import MysqlCommand
from ReadersWriters._MysqlConnection import MysqlConnection
from ReadersWriters._PickleSerialised import PickleSerialised
from ReadersWriters._TextFile import TextFile
from ReadersWriters._MysqlQueries import MysqlQueries

PandasDataFrame = TypeVar('DataFrame')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class ReadersWriters:
    @staticmethod
    def exists_serialised(path: str,
                          title: str,
                          ext: str) -> bool:
        """Check if the serialised object exists.
        :param path: the output directory path.
        :param title: the title of the output file.
        :param ext: the extension of the output file.
        :return: indicates if the file exists.
        """
        reader = PickleSerialised()
        reader.set(path, title, ext)
        return reader.exists()

    @staticmethod
    def exists_mysql(db_schema: str,
                     db_table: str) -> bool:
        """Check if MySQL table exists.
        :param db_schema: the MySQL database schema.
        :param db_table: the MySQL table.
        :return: indicates if the file exists.
        """
        queries = MysqlQueries()
        query = queries.exists_table(db_table)
        output = ReadersWriters.load_mysql_query(query, db_schema, dataframing=True)
        return output is not None and len(output) > 0

    @staticmethod
    def exists_csv(path: str,
                   title: str,
                   ext: str= "csv") -> bool:
        """Check if the CSV file exists.
        :param path: the directory path of the CSV file.
        :param title: the file name of the CSV file.
        :param ext: the extension of the CSV file (default: 'csv').
        :return: indicates if the file exists.
        """
        reader = CsvFile()
        reader.set(path, title, ext)
        return reader.exists()

    @staticmethod
    def exists_text(path: str,
                    title: str,
                    ext: str) -> bool:
        """Check if the text file exists.
        :param path: the directory path of the text file.
        :param title: the file name of the text file.
        :param ext: he extension of the text file.
        :return: indicates if the file exists.
        """
        reader = TextFile()
        reader.set(path, title, ext)
        return reader.exists()

    @staticmethod
    def exists_mysql_column(db_schema: str,
                            db_table: str,
                            column: str) -> bool:
        """Check if a column exists in a MySQL table.
        :param db_schema: the MySQL database schema.
        :param db_table: the MySQL table.
        :param column: name of the column.
        :return: indicates if the column exists.
        """
        queries = MysqlQueries()
        query = queries.exists_column(db_table, column)
        output = ReadersWriters.load_mysql_query(query, db_schema, dataframing=True)
        return output is not None and len(output) > 0

    @staticmethod
    def exists_csv_column(path: str,
                          title: str,
                          column: str,
                          ext: str) -> bool:
        """Check if the CSV file exists.
        :param path: the directory path of the CSV file.
        :param title: the file name of the CSV file.
        :param column: name of the column.
        :param ext: the extension of the CSV file (default: 'csv').
        :return: indicates if the column exists.
        """
        reader = CsvFile()
        reader.set(path, title, ext)
        return reader.exists_column(column)

    @staticmethod
    def size_serialised(path: str,
                        title: str,
                        ext: str) -> int:
        """Check the size of the saved file.
        :param path: the directory path of the serialised file.
        :param title: the title of the output file.
        :param ext: the extension of the output file.
        :return: showing stat information of the file.
        """
        reader = PickleSerialised()
        reader.set(path, title, ext)
        return reader.size()

    @staticmethod
    def size_mysql_table(db_schema: str,
                         db_table: str) -> int:
        """Check number of records in a MySQL table.
        :param db_schema: the MySQL database schema.
        :param db_table: the MySQL table.
        :return: number of rows in the table.
        """
        output = ReadersWriters.load_mysql_table(db_schema, db_table, dataframing=True)
        return output is not None if int(output[0][0]) else 0

    @staticmethod
    def size_mysql_query(query: str,
                         db_schema: str) -> int:
        """Check number of records in the output result of the MySQL query.
        :param query: the mysql query to execute.
        :param db_schema: the MySQL database schema.
        :return: number of rows in the output table.
        """
        output = ReadersWriters.load_mysql_query(query, db_schema, dataframing=True)
        return output is not None if int(output[0][0]) else 0

    @staticmethod
    def size_csv(path: str,
                 title: str,
                 ext: str= "csv") -> int:
        """Check number of lines in the CSV file.
        :param path: the directory path of the CSV file.
        :param title: the title of the CSV file.
        :param ext: the extension of the CSV file.
        :return: number of lines in the CSV file.
        """
        reader = CsvFile()
        reader.set(path, title, ext)
        return reader.size()

    @staticmethod
    def size_text(path: str,
                  title: str,
                  ext: str) -> int:
        """Check number of lines in the text file.
        :param path: the directory path of the text file.
        :param title: the title of the text file.
        :param ext: the extension of the text file.
        :return: number of lines in the text file.
        """
        reader = TextFile()
        reader.set(path, title, ext)
        return reader.size()

    @staticmethod
    def save_serialised(path: str,
                        title: str,
                        objects: Any):
        """Serialise the object (Pickle protocol=4), without compression.
        :param path: the directory path of the serialised file.
        :param title: the title of the output file.
        :param objects: the object to be saved.
        """
        writer = PickleSerialised()
        writer.set(path, title, "pickle")
        writer.save(objects)

    @staticmethod
    def save_serialised_compressed(path: str,
                                   title: str,
                                   objects: Any):
        """Serialise the object (Pickle protocol=4), then compress (BZ2 compression).
        :param path: the directory path of the serialised file.
        :param title: the title of the output file.
        :param objects: the object to be saved.
        """
        writer = PickleSerialised()
        writer.set(path, title, "bz2")
        writer.save_bz2(objects)

    @staticmethod
    def save_mysql(query: str,
                   data: Callable[[List, PandasDataFrame], None],
                   db_schema: str,
                   db_table: str,
                   batch: int=None):
        """Write several rows of data into a MySQL table, using dataframe or list.
        :param query: the mysql query to execute, if data type is list.
        :param data: the mysql query to execute (applicable for data type of list).
        :param db_schema: the MySQL database schema.
        :param db_table: the MySQL table.
        :param batch: indicates if data to be written in batches.
        """
        db = MysqlConnection()
        db.set(db_schema)
        engine = db.open()
        dbc = MysqlCommand(engine, db.db_session_vars)
        if data is None:
            dbc.write(query)
        else:
            if batch is None:
                dbc.write_many(query, data, db_schema, db_table, batch_title="all")
            else:
                size = int(math.ceil(len(data) / batch))
                for i in range(0, size):
                    dbc.write_many(query, data[i * batch:(i + 1) * batch], db_schema, db_table,
                                   batch_title=str([i * batch, (i + 1) * batch]))
        db.close()

    @staticmethod
    def save_csv(path: str,
                 title: str,
                 data: Callable[[List, Dict, PandasDataFrame], None],
                 append: bool=False,
                 ext: str= "csv",
                 **kwargs: Any):
        """Append to CSV file using dataframe, dictionary or list.
        :param path: the directory path of the CSV file.
        :param title: the file name of the CSV file.
        :param data: the data to write.
        :param append: indicates if data to be appended to an existing file.
        :param ext: the extension of the CSV file (default: 'csv').
        :param kwargs: any other arguments that the selected writer may accept.
        """
        writer = CsvFile()
        writer.set(path, title, ext)
        if append is False:
            writer.reset()
        writer.append(data, **kwargs)

    @staticmethod
    def save_text(path: str,
                  title: str,
                  data: Callable[[List, Dict, PandasDataFrame], None],
                  append=False,
                  ext="log"):
        """Append to text file using dataframe, dictionary or list.
        :param path: the directory path of the text file.
        :param title: the file name of the text file.
        :param data: the data to write.
        :param append: indicates if data to be appended to an existing file.
        :param ext: the extension of the CSV file.
        """
        writer = TextFile()
        writer.set(path, title, ext)
        if append is False:
            writer.reset()
        writer.append(data)

    @staticmethod
    def load_serialised(path: str,
                        title: str) -> Any:
        """Load a serialised object, that was not compressed.
        :param path: the directory path of the serialised file.
        :param title: the title of the output file.
        :return: the loaded python object.
        """
        reader = PickleSerialised()
        reader.set(path, title, "pickle")
        return reader.load()

    @staticmethod
    def load_serialised_compressed(path: str,
                                   title: str) -> Any:
        """Load a serialised object, that was compressed (BZ2 compression).
        :param path: the directory path of the serialised file.
        :param title: the title of the output file.
        :return: the loaded python object.
        """
        reader = PickleSerialised()
        reader.set(path, title, "bz2")
        return reader.load_bz2()

    @staticmethod
    def load_mysql_table(db_schema: str,
                         db_table: str,
                         dataframing: bool=True):
        """"Read a MySQL table and return it as a dataframe or a list.
        :param db_schema: the MySQL database schema.
        :param db_table: the MySQL table.
        :param dataframing: indicates if the return data will be dataframe or list.
        :return: the output of the executed query (read table).
        """
        query = "SELECT * FROM " + db_table
        return ReadersWriters.load_mysql_query(query, db_schema, dataframing)

    @staticmethod
    def load_mysql_query(query: str,
                         db_schema: str,
                         dataframing: bool=True,
                         batch: int=None,
                         float_round_vars: List=None,
                         float_round: int=None) -> Callable[[List, PandasDataFrame], None]:
        """Execute a MySQL query and return the output result as dataframe or list.
        :param query: the mysql query to execute.
        :param db_schema: the MySQL database schema.
        :param dataframing: indicates if the return data will be dataframe or list.
        :param batch: indicates if data is loaded batch by batch.
        :param float_round_vars: list of float variables that needs to be rounded.
        :param float_round: the rounding precision for the 'float_round_vars' option.
        :return: the output of the executed query.
        """
        db = MysqlConnection()
        db.set(db_schema)
        engine = db.open()
        dbc = MysqlCommand(engine, db.db_session_vars)
        output = dbc.read(query, dataframing, batch, float_round_vars, float_round)
        db.close()
        return output

    @staticmethod
    def load_csv(path: str,
                 title: str,
                 skip: int=0,
                 dataframing: bool=True,
                 ext: str= "csv",
                 **kwargs: Any) -> Callable[[List, PandasDataFrame], None]:
        """Read the CSV file into dataframe or list.
        :param path: the directory path of the CSV file.
        :param title: the file name of the CSV file.
        :param skip: lines to skip before reading or writing.
        :param dataframing: indicates if the outputs must be saved into dataframe.
        :param ext: the extension of the CSV file (default: 'csv').
        :param kwargs: any other arguments that the selected reader may accept.
        :return: the read file contents.
        """
        reader = CsvFile()
        reader.set(path, title, ext)
        return reader.read(skip, dataframing, **kwargs)

    @staticmethod
    def load_text(path: str,
                  title: str,
                  ext: str,
                  skip: int) -> List:
        """Read the text file into dataframe or list.
        :param path: the directory path of the text file.
        :param title: the file name of the text file.
        :param ext: the extension of the text file.
        :param skip: lines to skip before reading or writing.
        """
        reader = TextFile()
        reader.set(path, title, ext)
        return reader.read(skip)

    @staticmethod
    def load_mysql_procedure(db_procedure: str,
                             args: List,
                             db_schema: str) -> Any:
        """Execute a MySQL procedure, and return the raw output results (if applicable).
        :param db_procedure: the mysql procedure to execute.
        :param args: the procedure's input arguments.
        :param db_schema: the MySQL database schema.
        :return: the raw output results (if applicable).
        """
        db = MysqlConnection()
        db.set(db_schema)
        engine = db.open()
        dbc = MysqlCommand(engine, db.db_session_vars)
        output = dbc.call_proc(db_procedure, args)
        db.close()
        return output

    @staticmethod
    def reset_mysql_table(db_schema: str,
                          db_table: str):
        """Truncate the MySQL table.
        :param db_schema: the MySQL database schema.
        :param db_table: the MySQL table.
        """
        query = "TRUNCATE TABLE " + db_table
        ReadersWriters.load_mysql_query(query, db_schema)

    @staticmethod
    def reset_csv(path: str,
                  title: str,
                  ext: str= "csv"):
        """Reset the CSV file reader/writer.
        :param path: the directory path of the CSV file.
        :param title: the file name of the CSV file.
        :param ext: the extension of the CSV file (default: 'csv').
        """
        reader = CsvFile()
        reader.set(path, title, ext)
        reader.reset()

    @staticmethod
    def reset_text(path: str,
                   title: str,
                   ext: str="log"):
        """Reset the text file reader/writer.
        :param path: the directory path of the text file.
        :param title: the file name of the text file.
        :param ext: the extension of the text file.
        """
        reader = TextFile()
        reader.set(path, title, ext)
        reader.reset()

    @staticmethod
    def mysql_query_create(table: str,
                           columns: List,
                           col_types: List,
                           defaults: List,
                           primary_keys: List=list(),
                           unique_keys: List=list(),
                           meta: str="ENGINE=InnoDB DEFAULT CHARSET=latin1") -> str:
        """Generate a bespoke create query.
        :param table: the MySQL table.
        :param columns: the list of columns.
        :param col_types: the list of column types for columns.
        :param defaults: the list of default values for columns.
        :param primary_keys: the list of primary keys for the table.
        :param unique_keys: the list of unique keys for the table.
        :param meta: the table metadata (Engine and charset).
        :return: the MySQL query
        """
        queries = MysqlQueries()
        return queries.create(table, columns, col_types, defaults, primary_keys, unique_keys, meta)

    @staticmethod
    def mysql_query_insert(table: str,
                           columns: List,
                           col_types: List) -> str:
        """Generate a bespoke insert query.
        :param table: the MySQL table.
        :param columns: the list of columns.
        :param col_types: the list of column types for columns.
        :return: the MySQL query
        """
        queries = MysqlQueries()
        return queries.insert(table, columns, col_types)

    @staticmethod
    def mysql_query_drop(table: str) -> str:
        """Generate a bespoke drop query.
        :param table: the MySQL table.
        :return: the MySQL query
        """
        queries = MysqlQueries()
        return queries.drop(table)

    @staticmethod
    def question_overwrite(name: str) -> bool:
        """Commandline question, which aims to get confirmation of the user.
        :param name: the subject name, which needs confirmation.
        :return: the user's response.
        """
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
    def create_directories(path: str):
        """Create directories in the specified path if do not exist.
        :param path: the directory path to a file.
        """
        if not os.path.exists(path):
            try:
                os.makedirs(path, exist_ok=True)
            except OSError as e:
                sys.exit(__name__ + ' - Directory creation error: {0:d}:\n {1:s}'.format(e.args[0], str(e.args[1])))
