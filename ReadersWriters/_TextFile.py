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
""" It is an interface for reading and writing text files.
"""

from typing import Dict, TypeVar, Callable
import os
import sys
from typing import List
from collections import OrderedDict
import pandas as pd
import numpy as np
import pprint as pp
import logging
from Configs.CONSTANTS import CONSTANTS

PandasDataFrame = TypeVar('DataFrame')
CollectionsOrderedDict = TypeVar('OrderedDict')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class TextFile:
    def __init__(self,
                 max_width: int=100000000):
        """Initialise the objects and constants.
        """
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__path = None
        self.__max_width = max_width

    def set(self,
            path: str,
            title: str,
            ext: str):
        """Set the text file for reading or writing.
        :param path: the directory path of the text file.
        :param title: the file name of the text file.
        :param ext: the extension of the text file.
        """
        self.__logger.debug("Set the text file.")
        self.__path = os.path.join(path, title + "." + ext)

    def reset(self):
        """Reset the text file reader/writer.
        """
        self.__logger.debug("Reset the text file.")
        try:
            open(self.__path, 'w').close()
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: \n" + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Could not create the file: \n" + self.__path)
            sys.exit()

    def exists(self) -> bool:
        """Check if the text file exists.
        :return: indicates if the file exists.
        """
        self.__logger.debug("Check if the text file exists.")
        return os.path.isfile(self.__path)

    def read(self,
             skip: int) -> List:
        """Read the text file into list.
        :param skip: lines to skip before reading.
        :return: the read file contents.
        """
        self.__logger.debug("Read the text file.")
        rows = self.__read_array(skip)
        return rows

    def __read_array(self,
                     skip: int) -> List:
        """Read the text file into array.
        :param skip: lines to skip before reading.
        :return: the read file contents.
        """
        self.__logger.debug("Read the text file into array.")
        rows = []
        i = 0
        try:
            with open(self.__path, "r") as f:
                for line in f:
                    i += 1
                    if i > skip:
                        rows.append(line)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: \n" + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not read the Text file: \n" + self.__path)
            sys.exit()
        return rows

    def append(self,
               data: Callable[[List, Dict, PandasDataFrame], None]):
        """Append to text file using dataframe, dictionary or list.
        :param data: the data to write.
        """
        self.__logger.debug("Append to the text file.")
        # Set numpy writing options
        np.set_printoptions(threshold=sys.maxsize)
        # Set Pandas writing options
        pd.set_option("display.width", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)

        if isinstance(data, pd.DataFrame):
            self.__append_dataframe(data)
        elif isinstance(data, list) or isinstance(data, str):
            self.__append_pretty(data)
        elif isinstance(data, dict) or isinstance(data, OrderedDict):
            self.__append_pretty_dict(data)
        else:
            self.__logger.error(__name__ + " - Invalid object to write into Text file: \n" + str(type(data)))
            sys.exit()

        # Reset numpy writing options
        np.set_printoptions(threshold=None)
        # Reset Pandas writing options
        pd.reset_option("display.width")
        pd.reset_option("display.max_rows")
        pd.reset_option("display.max_columns")

    def __append_dataframe(self,
                           data: PandasDataFrame,
                           label: str = ''):
        """Append to text file using dataframe.
        :param data: the dataframe to write.
        """
        try:
            with open(self.__path, 'a') as f:
                pp.pprint(label + ":", stream=f)
                data.to_string(f, columns=data.columns.values, header=True, index=True)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: \n" + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not append dataframe to file: \n" + self.__path)
            sys.exit()

    def __append_pretty(self,
                        data: Callable[[List, str], None]):
        """Append to text file using list.
        :param data: the data to write.
        """
        self.__logger.debug("Append a Dataframe to the text file.")
        try:
            with open(self.__path, 'a') as f:
                pp.pprint(data, stream=f, width=self.__max_width, depth=self.__max_width, compact=True)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: \n" + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not append dictionary to file: \n" + self.__path)
            sys.exit()

    def __append_pretty_dict(self,
                             data: Callable[[Dict, CollectionsOrderedDict], None],
                             label: str = ''):
        """Append to text file using dictionary.
        :param data: the data to write.
        """
        self.__logger.debug("Append a Dictionary to the text file.")
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
            self.__logger.error(__name__ + " - Can not open the file: \n" + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not append dictionary to file: \n" + self.__path)
            sys.exit()

    def size(self) -> int:
        """Check number of lines in the text file.
        :return: number of lines in the file.
        """
        self.__logger.debug("Check number of lines in the CSV file.")
        cnt_lines = 0
        try:
            with open(self.__path, "r") as f:
                for _ in f:
                    cnt_lines += 1
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: \n" + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not read the Text file: \n" + self.__path)
            sys.exit()
        return cnt_lines
