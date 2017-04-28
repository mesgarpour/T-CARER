#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from sqlalchemy.orm import sessionmaker
import pandas as pd
import pandas.io.sql as pds
import sys
import warnings
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


class MysqlCommand:
    """Execute Mysql commands"""

    def __init__(self, connection, db_session_vars):
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__engine = connection
        self.__db_session_vars = db_session_vars

    def read(self, query, dataframing=True, batch=None, float_round_vars=None, float_round=None):
        self.__logger.debug(__name__)
        self.__logger.info("Reading")
        if dataframing:
            result = self.__read_df(query, batch, float_round_vars, float_round)
        else:
            result = self.__read_arr(query)
        return result

    def __read_df(self, query, batch=None, float_round_vars=None, float_round=None):
        """Execute Reader and return a dataframe"""
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
                        self.__logger.info("... " + str(step))
                        if float_round is not None:
                            for col in float_round_vars:
                                if col in df:
                                    df[col] = df[col].astype(float).round(float_round)
                        if result is None:
                            result = df
                        else:
                            result = result.append(df, ignore_index=True)
        except():
            self.__logger.error(__name__ + " - DB read execute related error!\n" + str(sys.exc_info()[0]))
            sys.exit()
        finally:
            if session is not None:
                session.close()
            if conn is not None:
                conn.close()

        # replace numpy nan with None
        result = result.where(pd.notnull(result), None)
        return result

    def __read_arr(self, query):
        """Execute Reader and return an array"""
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
            self.__logger.error(__name__ + " - DB read execute related error: {0:s}".format(str(e.args[0])))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - DB read execute related error!\n" + str(sys.exc_info()[0]))
            sys.exit()
        finally:
            if session is not None:
                session.close()
            if conn is not None:
                conn.close()
        return result

    def call_proc(self, query, args):
        """Execute Reader and return an array"""
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
                cursor.callproc(query, args)
                result = list(cursor.fetchall())
                conn.commit()
        except Exception as e:
            self.__logger.error(__name__ + " - DB read execute related error: {0:s}".format(str(e.args[0])))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - DB read execute related error!\n" + str(sys.exc_info()[0]))
            sys.exit()
        finally:
            if session is not None:
                session.close()
            if conn is not None:
                cursor.close()
                conn.close()
        return result

    def write(self, query):
        """Execute array Writer"""
        self.__logger.debug(__name__)
        self.__logger.info("Writing")
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
            self.__logger.error(__name__ + " - DB Cursor execute related error: {0:s}".format(str(e.args[0])))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - DB Cursor execute related error!\n" + str(sys.exc_info()[0]))
            sys.exit()
        finally:
            if session is not None:
                session.close()
            if conn is not None:
                conn.close()

    def write_many(self, query, data, schema, table):
        """Execute Writer Many"""
        self.__logger.debug(__name__)
        self.__logger.info("Writing multiple rows into MySQL")
        if isinstance(data, pd.DataFrame):
            self.__write_many_df(data, schema, table)
        elif isinstance(data, list):
            self.__write_many_arr(query, data)
        else:
            self.__logger.error(__name__ + " - Invalid object to write into MySQL table!\n" + str(type(data)))
            sys.exit()

    def __write_many_df(self, data, schema, table, if_exists='append'):
        """Execute dataframe Writer Many"""
        session = None
        conn = None

        try:
            with warnings.catch_warnings():  # suppress warnings
                warnings.simplefilter("ignore")
                conn = self.__engine.connect()

                # Open the session
                session = sessionmaker(bind=self.__engine, autoflush=True, autocommit=True)()
                self.__set_session_vars(session)

                data.to_sql(schema=schema,
                            name=table,
                            con=conn,
                            flavor='mysql',
                            if_exists=if_exists,
                            index=False)
        except Exception as e:
            self.__logger.error(__name__ + " - DB cursor execute-many related error: {0:d}".format(str(e.args[0])))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - DB cursor execute-many related error!\n" + str(sys.exc_info()[0]))
            sys.exit()
        finally:
            if session is not None:
                session.close()
            if conn is not None:
                conn.close()

    def __write_many_arr(self, query, data):
        """Execute array Writer Many"""
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
            self.__logger.error(__name__ + " - DB cursor execute-many related error: {0:s}".format(str(e.args[0])))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - DB cursor execute-many related error!\n" + str(sys.exc_info()[0]))
            sys.exit()
        finally:
            if session is not None:
                session.close()
            if conn is not None:
                conn.close()
        
    def __set_session_vars(self, session):
        """Set session variables"""
        for query in self.__db_session_vars:
            session.execute(query)
