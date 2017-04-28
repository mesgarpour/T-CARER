#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import pandas as pd
import os
import csv
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


class _CsvFile:
    """Read from or Write to a csv file
    """

    def __init__(self):
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__path = None
        self.__line_width = 100000000

    def set(self, path, title, extension="csv"):
        self.__logger.debug(__name__)
        self.__path = os.path.join(path, title + "." + extension)

    def reset(self):
        try:
            open(self.__path, 'w').close()
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Could not create the file: " + self.__path)
            sys.exit()
            
    def exists(self):
        return os.path.isfile(self.__path)

    def exists_column(self, column, skip=0):
        i = 0
        try:
            with open(self.__path, "r") as f:
                for line in f:
                    if i > skip:
                        if column not in set(line.split(",")):
                            return False
                        else:
                            return True
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not read the file: " + self.__path)
            sys.exit()

    def read(self, skip, dataframing=True, **kwargs):
        self.__logger.debug(__name__)
        if dataframing:
            rows = self.__read_dataframe(skip, **kwargs)
        else:
            rows = self.__read_array(skip)
        return rows

    def __read_dataframe(self, skip, **kwargs):
        try:
            rows = pd.read_csv(self.__path, skiprows=skip, **kwargs)
        except():
            self.__logger.error(__name__ + " - Can not read the file into a dataframe: " + self.__path)
            sys.exit()
        return rows

    def __read_array(self, skip):
        rows = []
        i = 0
        with open(self.__path, "r") as f:
            try:
                for line in f:
                    i += 1
                    if i > skip:
                        rows.append(line.split(","))
            except (OSError, IOError) as e:
                self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n" + str(e))
                sys.exit()
            except():
                self.__logger.error(__name__ + " - Can not read the file: " + self.__path)
                sys.exit()
        return rows

    def append(self, data, **kwargs):
        self.__logger.debug(__name__)
        if isinstance(data, pd.DataFrame):
            self.__append_dataframe(data, **kwargs)
        elif isinstance(data, list):
            self.__append_array(data)
        elif isinstance(data, dict):
            self.__append_dict(data)
        else:
            self.__logger.error(__name__ + " - Invalid object to write into file!\n" + str(type(data)))
            sys.exit()
        return True

    def __append_dataframe(self, data, **kwargs):
        kwargs["header"] = False if "header" not in kwargs.keys() else kwargs["header"]
        kwargs["index"] = False if "index" not in kwargs.keys() else kwargs["index"]
        try:
            with open(self.__path, 'a') as f:
                pd.set_option("display.width", self.__line_width)
                pd.set_option("display.max_rows", data.shape[0])
                pd.set_option("display.max_columns", data.shape[1])
                data.to_csv(f, header=kwargs["header"], index=kwargs["index"])
                pd.reset_option("display.width")
                pd.reset_option("display.max_rows")
                pd.reset_option("display.max_columns")
        except():
            self.__logger.error(__name__ + " - Can not append dataframe to file: " + self.__path)
            sys.exit()

    def __append_dict(self, data):
        try:
            with open(self.__path, 'a') as f:
                w = csv.DictWriter(f, data.keys())
                w.writeheader()
                w.writerow(data)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not append dictionary to file: " + self.__path)
            sys.exit()

    def __append_array(self, data):
        # create 2D list
        if data is None or data == "" or data == []:
            return
        elif not isinstance(data, list):
            data = [[data]]
        elif not isinstance(data[0], list):
            data = [data]

        # write
        try:
            with open(self.__path, 'a+b') as f:
                # flatten 2D list
                for row in data:
                    f.write((",".join(row) + "\n").encode())
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not write a row into the file: " + self.__path)
            sys.exit()

    def size(self):
        self.__logger.debug(__name__)
        cnt_lines = 0
        try:
            with open(self.__path, "r") as f:
                for _ in f:
                    cnt_lines += 1
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not read the file: " + self.__path)
            sys.exit()
        return cnt_lines
