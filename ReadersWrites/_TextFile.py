#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import pandas as pd
import os
import numpy as np
import pprint as pp
from collections import OrderedDict
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


class TextFile:
    """Read from or Write to a text file
    """

    def __init__(self):
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__path = None
        self.__max_width = 100000000

    def set(self, path, title, extension):
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
        
    def read(self, skip):
        self.__logger.debug(__name__)
        rows = self.__read_array(skip)
        return rows

    def __read_array(self, skip):
        rows = []
        i = 0
        try:
            with open(self.__path, "r") as f:
                for line in f:
                    i += 1
                    if i > skip:
                        rows.append(line)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not read the Text file: " + self.__path)
            sys.exit()
        return rows

    def append(self, data):
        self.__logger.debug(__name__)
        # numpy
        np.set_printoptions(threshold=sys.maxsize)
        # Pandas
        pd.set_option("display.width", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)

        if isinstance(data, pd.DataFrame):
            self.__append_dataframe(data, "")
        elif isinstance(data, list) or isinstance(data, str):
            self.__append_pretty(data)
        elif isinstance(data, dict) or isinstance(data, OrderedDict):
            self.__append_pretty_dict(data, "")
        else:
            self.__logger.error(__name__ + " - Invalid object to write into Text file!\n" + str(type(data)))
            sys.exit()

        # numpy
        np.set_printoptions(threshold=None)
        # Pandas
        pd.reset_option("display.width")
        pd.reset_option("display.max_rows")
        pd.reset_option("display.max_columns")
        return True

    def __append_dataframe(self, data, label):
        try:
            with open(self.__path, 'a') as f:
                pp.pprint(label + ":", stream=f)
                data.to_string(f, columns=data.columns.values, header=True, index=True)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not append dataframe to file: " + self.__path)
            sys.exit()

    def __append_pretty(self, data):
        try:
            with open(self.__path, 'a') as f:
                pp.pprint(data, stream=f, width=self.__max_width, depth=self.__max_width, compact=True)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not append dictionary to file: " + self.__path)
            sys.exit()

    def __append_pretty_dict(self, data, label):
        try:
            for label, value in data.items():
                if isinstance(value, dict) or isinstance(value, OrderedDict):
                    self.__append_pretty_dict(value, label)
                elif isinstance(value, pd.DataFrame):
                    self.__append_dataframe(value, "")
                else:
                    with open(self.__path, 'a') as f:
                        pp.pprint(label + ":", stream=f)
                        pp.pprint(value, stream=f, width=self.__max_width, depth=self.__max_width, compact=True)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not append dictionary to file: " + self.__path)
            sys.exit()

    def size(self):
        cnt_lines = 0
        try:
            with open(self.__path, "r") as f:
                for _ in f:
                    cnt_lines += 1
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not read the Text file: " + self.__path)
            sys.exit()
        return cnt_lines
