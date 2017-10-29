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
""" It is an interface for reading and writing Comma Separated Values (CSV) files.
"""

from typing import Dict, List, TypeVar, Any, Callable
import sys
import pandas as pd
import os
import csv
import logging
from Configs.CONSTANTS import CONSTANTS

PandasDataFrame = TypeVar('DataFrame')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class CsvFile:
    def __init__(self):
        """Initialise the objects and constants.
        """
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__path = None

    def set(self,
            path: str,
            title: str,
            ext: str = "csv"):
        """Set the CSV file for reading or writing.
        :param path: the directory path of the CSV file.
        :param title: the file name of the CSV file.
        :param ext: the extension of the CSV file (default: 'csv').
        """
        self.__logger.debug("Set the CSV file.")
        self.__path = os.path.join(path, title + "." + ext)

    def reset(self):
        """Reset the CSV file reader/writer.
        """
        self.__logger.debug("Reset the CSV File.")
        try:
            open(self.__path, 'w').close()
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Could not create the file: \n" + self.__path)
            sys.exit()
            
    def exists(self) -> bool:
        """Check if the CSV file exists.
        :return: indicates if the file exists.
        """
        self.__logger.debug("Check if the CSV file exists.")
        return os.path.isfile(self.__path)

    def exists_column(self,
                      column: str,
                      skip: int = 0) -> bool:
        """Check if the CSV file exists.
        :param column: name of the column.
        :param skip: lines to skip before reading or writing.
        :return: indicates if the column exists.
        """
        self.__logger.debug("Check if a column exists in the CSV File.")
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
            self.__logger.error(__name__ + " - Can not read the file: \n" + self.__path)
            sys.exit()

    def read(self,
             skip: int=0,
             dataframing: bool=True,
             **kwargs: Any) -> Callable[[List, PandasDataFrame], None]:
        """Read the CSV file into dataframe or list.
        :param skip: lines to skip before reading.
        :param dataframing: indicates if the outputs must be saved into dataframe.
        :param kwargs: any other arguments that the selected reader may accept.
        :return: the read file contents.
        """
        self.__logger.debug("Read the CSV File.")
        if dataframing:
            rows = self.__read_dataframe(skip, **kwargs)
        else:
            rows = self.__read_array(skip)
        return rows

    def __read_dataframe(self,
                         skip: int=0,
                         **kwargs: Any) -> PandasDataFrame:
        """Read the CSV file into dataframe.
        :param skip: lines to skip before reading.
        :param kwargs: any other arguments that the selected reader may accept.
        :return: the read file contents.
        """
        self.__logger.debug("Read the CSV File into Dataframe.")
        try:
            rows = pd.read_csv(self.__path, skiprows=skip, **kwargs)
        except():
            self.__logger.error(__name__ + " - Can not read the file into a dataframe: \n" + self.__path)
            sys.exit()
        return rows

    def __read_array(self,
                     skip: int = 0) -> List:
        """Read the CSV file into array.
        :param skip: lines to skip before reading.
        :return: the read file contents.
        """
        self.__logger.debug("Read the CSV File into array.")
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
                self.__logger.error(__name__ + " - Can not read the file: \n" + self.__path)
                sys.exit()
        return rows

    def append(self,
               data: Callable[[List, Dict, PandasDataFrame], None],
               **kwargs: Any):
        """Append to CSV file using dataframe, dictionary or list.
        :param data: the data to write.
        :param kwargs: any other arguments that the selected writer may accept.
        """
        self.__logger.debug("Append to the CSV file.")
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

    def __append_dataframe(self,
                           data: PandasDataFrame,
                           max_line_width: int = 100000000,
                           **kwargs: Any):
        """Append to CSV file using dataframe.
        :param data: the dataframe to write.
        :param max_line_width: max line width (PANDAS: display.width).
        :param kwargs: any other arguments that the selected writer may accept.
        """
        self.__logger.debug("Append a Dataframe to the CSV file.")
        kwargs["header"] = False if "header" not in kwargs.keys() else kwargs["header"]
        kwargs["index"] = False if "index" not in kwargs.keys() else kwargs["index"]
        try:
            with open(self.__path, 'a') as f:
                pd.set_option("display.width", max_line_width)
                pd.set_option("display.max_rows", data.shape[0])
                pd.set_option("display.max_columns", data.shape[1])
                data.to_csv(f, header=kwargs["header"], index=kwargs["index"])
                pd.reset_option("display.width")
                pd.reset_option("display.max_rows")
                pd.reset_option("display.max_columns")
        except():
            self.__logger.error(__name__ + " - Can not append dataframe to file: \n" + self.__path)
            sys.exit()

    def __append_dict(self,
                      data: Dict[str, str]):
        """Append to CSV file using dictionary.
        :param data: the dictionary to write.
        """
        self.__logger.debug("Append a Dictionary to the CSV file.")
        try:
            with open(self.__path, 'a') as f:
                w = csv.DictWriter(f, data.keys())
                w.writeheader()
                w.writerow(data)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n" + str(e))
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Can not append dictionary to file: \n" + self.__path)
            sys.exit()

    def __append_array(self,
                       data: List):
        """Append to CSV file using list.
        :param data: the list to write.
        """
        self.__logger.debug("Append an Array to the CSV file.")
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
            self.__logger.error(__name__ + " - Can not write a row into the file: \n" + self.__path)
            sys.exit()

    def size(self) -> int:
        """Check number of lines in the CSV file.
        :return: number of lines in the file.
        """
        self.__logger.debug("Check number of lines in the CSV file.")
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
