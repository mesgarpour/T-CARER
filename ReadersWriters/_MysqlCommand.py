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
"""It is an interface for executing MySQL queries using the SQLAlchemy connection.
"""

from typing import List, TypeVar, Callable
from sqlalchemy.orm import sessionmaker
import pandas as pd
import pandas.io.sql as pds
import sys
import warnings
import logging
from Configs.CONSTANTS import CONSTANTS

PandasDataFrame = TypeVar('DataFrame')
SqlalchemyEngine = TypeVar('Engine')
SqlAlchemySessionMaker = TypeVar('sessionmaker')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class MysqlCommand:
    def __init__(self,
                 connection: SqlalchemyEngine,
                 db_session_vars: List):
        """Initialise the objects and constants.
        :param connection: the SQLAlchemy Engine. The Engine is the starting point for any SQLAlchemy application.
        :param db_session_vars: session variables that will used before execution of queries.
        """
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__engine = connection
        self.__db_session_vars = db_session_vars

    def read(self,
             query: str,
             dataframing: bool=True,
             batch: int=None,
             float_round_vars: List=None,
             float_round: int=None) -> Callable[[List, PandasDataFrame], None]:
        """Execute a MySQL query and return the output result as dataframe or list.
        :param query: the mysql query to execute.
        :param dataframing: indicates if the return data will be dataframe or list.
        :param batch: indicates if data is loaded batch by batch (available for dataframe type).
        :param float_round_vars: list of float variables that needs to be rounded (available for dataframe type).
        :param float_round: the rounding precision for the 'float_round_vars' option (available for dataframe type).
        :return: the output of the executed query.
        """
        self.__logger.debug("Reading from MySQL database.")
        if dataframing:
            result = self.__read_df(query, batch, float_round_vars, float_round)
        else:
            result = self.__read_arr(query)
        return result

    def __read_df(self,
                  query: str,
                  batch: int=None,
                  float_round_vars: List=None,
                  float_round: int=None) -> PandasDataFrame:
        """Execute a MySQL query and return the output result as dataframe.
        :param query: the mysql query to execute.
        :param batch: indicates if data is loaded batch by batch.
        :param float_round_vars: list of float variables that needs to be rounded.
        :param float_round: the rounding precision for the 'float_round_vars' option.
        :return: the output of the executed query.
        """
        self.__logger.debug("Reading from MySQL database and outputting into dataframe.")
        result = None
        step = 0
        session = None
        conn = None

        try:
            with warnings.catch_warnings():  # suppress warnings
                warnings.simplefilter("ignore")
                conn = self.__engine.connect()

                # Open the session
                session = sessionmaker(bind=self.__engine, autoflush=True, autocommit=True)()
                self.__set_session_vars(session)

                if batch is None:
                    result = pds.read_sql(sql=query, con=conn, coerce_float=False, chunksize=batch)
                else:
                    for df in pds.read_sql(sql=query, con=conn, coerce_float=False, chunksize=batch):
                        step += batch
                        self.__logger.info("Batch: " + str(step) + ".")
                        if float_round_vars is not None:
                            for col in float_round_vars:
                                if col in df:
                                    df[col] = df[col].astype(float).round(float_round)
                        if result is None:
                            result = df
                        else:
                            result = result.append(df, ignore_index=True)
        except():
            self.__logger.error(__name__ + " - DB read execute related error: \n" + str(sys.exc_info()[0]))
            sys.exit()
        finally:
            if session is not None:
                session.close()
            if conn is not None:
                conn.close()

        # replace numpy nan with None
        result = result.where(pd.notnull(result), None)
        return result

    def __read_arr(self,
                   query: str) -> List:
        """Execute a MySQL query and return the output result as list.
        :param query: the mysql query to execute.
        :return: the output of the executed query.
        """
        self.__logger.debug("Reading from MySQL database and outputting into array.")
        session = None
        conn = None

        try:
            with warnings.catch_warnings():  # suppress warnings
                warnings.simplefilter("ignore")
                conn = self.__engine.connect()

                # Open the session
                session = sessionmaker(bind=self.__engine, autoflush=True, autocommit=True)()
                self.__set_session_vars(session)

                result = session.execute(query)
        except Exception as e:
            self.__logger.error(__name__ + " - DB read execute related error: \n{0:s}.".format(str(e.args[0])))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - DB read execute related error: \n" + str(sys.exc_info()[0]))
            sys.exit()
        finally:
            if session is not None:
                session.close()
            if conn is not None:
                conn.close()
        return result

    def call_proc(self,
                  db_procedure: str,
                  args: List) -> List:
        """Execute a MySQL procedure, and return the raw output results (if applicable).
        :param db_procedure: the mysql procedure to execute.
        :param args: the procedure's input arguments.
        :return: the raw output results (if applicable).
        """
        self.__logger.debug("Call MySQL procedure.")
        session = None
        cursor = None
        conn = None

        try:
            with warnings.catch_warnings():  # suppress warnings
                warnings.simplefilter("ignore")
                conn = self.__engine.raw_connection()
                cursor = conn.cursor()

                # Open the session
                session = sessionmaker(bind=self.__engine, autoflush=True, autocommit=True)()
                self.__set_session_vars(session)

                # execute
                cursor.callproc(db_procedure, args)
                result = list(cursor.fetchall())
                conn.commit()
        except Exception as e:
            self.__logger.error(__name__ + " - DB read execute related error: \n{0:s}.".format(str(e.args[0])))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - DB read execute related error: \n" + str(sys.exc_info()[0]))
            sys.exit()
        finally:
            if session is not None:
                session.close()
            if conn is not None:
                cursor.close()
                conn.close()
        return result

    def write(self,
              query: str):
        """Execute a MySQL query to write data into a MySQL table.
        :param query: the mysql query to execute.
        """
        self.__logger.debug("Write into MySQL database.")
        session = None
        conn = None

        try:
            with warnings.catch_warnings():  # suppress warnings
                warnings.simplefilter("ignore")
                conn = self.__engine.raw_connection()

                # Open the session
                session = sessionmaker(bind=self.__engine, autoflush=True, autocommit=True)()
                self.__set_session_vars(session)

                session.execute(query)
        except Exception as e:
            self.__logger.error(__name__ + " - DB Cursor execute related error: \n{0:s}.".format(str(e.args[0])))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - DB Cursor execute related error! \n" + str(sys.exc_info()[0]))
            sys.exit()
        finally:
            if session is not None:
                session.close()
            if conn is not None:
                conn.close()

    def write_many(self,
                   query: str,
                   data: Callable[[List, PandasDataFrame], None],
                   db_schema: str,
                   db_table: str,
                   batch_title: str= ""):
        """Write several rows of data into a MySQL table, using dataframe or list.
        :param query: the mysql query to execute (applicable for data type of list).
        :param data: the data to write into the MySQL table (list or a dataframe).
        :param db_schema: the MySQL database schema.
        :param db_table: the MySQL table.
        :param batch_title: the title for the current batch of write
        """
        self.__logger.info("Write Many into MySQL database: " + batch_title)
        if isinstance(data, pd.DataFrame):
            self.__write_many_df(data, db_schema, db_table)
        elif isinstance(data, list):
            self.__write_many_arr(query, data)
        else:
            self.__logger.error(__name__ + " - Invalid object to write into MySQL table: \n" + str(type(data)))
            sys.exit()

    def __write_many_df(self,
                        data: PandasDataFrame,
                        db_schema: str,
                        db_table: str,
                        if_exists: str = 'append'):
        """Execute 'many' MySQL queries to write data into a MySQL table, using dataframe.
        :param data: the data to write into the MySQL table.
        :param db_schema: the MySQL database schema.
        :param db_table: the MySQL table.
        :param if_exists: the dataframe write option ('fail', 'replace', or 'append').
        """
        self.__logger.debug("Write many into MySQL database, using Dataframe.")
        session = None
        conn = None

        try:
            with warnings.catch_warnings():  # suppress warnings
                warnings.simplefilter("ignore")
                conn = self.__engine.connect()

                # Open the session
                session = sessionmaker(bind=self.__engine, autoflush=True, autocommit=True)()
                self.__set_session_vars(session)

                data.to_sql(schema=db_schema,
                            name=db_table,
                            con=conn,
                            if_exists=if_exists,
                            index=False)
        except Exception as e:
            self.__logger.error(__name__ + " - DB cursor execute-many related error: {0:s}.".format(str(e.args[0])))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - DB cursor execute-many related error: \n" + str(sys.exc_info()[0]))
            sys.exit()
        finally:
            if session is not None:
                session.close()
            if conn is not None:
                conn.close()

    def __write_many_arr(self,
                         query: str,
                         data: List):
        """Execute 'many' MySQL queries to write data into a MySQL table, using list.
        :param query: the mysql query to execute.
        :param data: the data to write into the MySQL table.
        """
        self.__logger.debug("Write many into MySQL Database, using array.")
        session = None
        conn = None

        try:
            with warnings.catch_warnings():  # suppress warnings
                warnings.simplefilter("ignore")
                conn = self.__engine.connect()

                # Open the session
                session = sessionmaker(bind=self.__engine, autoflush=True, autocommit=True)()
                self.__set_session_vars(session)

                conn.execute(query, data)
        except Exception as e:
            self.__logger.error(__name__ + " - DB cursor execute-many related error: {0:s}.".format(str(e.args[0])))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - DB cursor execute-many related error: \n" + str(sys.exc_info()[0]))
            sys.exit()
        finally:
            if session is not None:
                session.close()
            if conn is not None:
                conn.close()

    def __set_session_vars(self,
                           session: SqlAlchemySessionMaker):
        """Set MySQL session variables.
        :param session: the session variables to set.
        """
        self.__logger.debug("Set Session Variables.")
        for query in self.__db_session_vars:
            session.execute(query)
