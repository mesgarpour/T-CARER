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
"""It is an interface for serialising python objects and optionally compressing them.
"""

from typing import TypeVar, Any
import pickle
import bz2
import sys
import os
import logging
from Configs.CONSTANTS import CONSTANTS

OsStatResult = TypeVar('stat_result')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class PickleSerialised:
    def __init__(self):
        """Initialise the objects and constants."""
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__path = None

    def set(self,
            path: str,
            title: str,
            ext: str):
        """Set the python serialiser configuration settings.
        :param path: the directory path of the serialised file.
        :param title: the title of the output file.
        :param ext: the extension of the output file.
        """
        self.__logger.debug("Set the pickle file.")
        self.__path = os.path.join(path, title + "." + ext)
        self.__logger.debug(self.__path)

    def exists(self) -> bool:
        """Check if the serialised object exists.
        :return: indicates if the file exists.
        """
        self.__logger.debug("Check if the pickle file exists.")
        return os.path.isfile(self.__path)

    def save(self,
             objects: Any):
        """Serialise the object (Pickle protocol=4), without compression.
        :param objects: the object to be saved.
        """
        self.__logger.debug("Save the Pickle file.")
        try:
            with open(self.__path, 'wb') as f:
                pickle.dump(objects, f, protocol=4)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: \n" + self.__path + "\n")
            print(e)
            sys.exit()
        except Exception as e:
            self.__logger.error(__name__ + " - Unable to save data to: \n" + self.__path + "\n")
            print(e)
            sys.exit()

        file_stats = os.stat(self.__path)
        self.__logger.info("Pickle Size: " + str(file_stats.st_size) + ".")

    def save_bz2(self,
                 objects: Any):
        """Serialise the object (Pickle protocol=4), then compress (BZ2 compression).
        :param objects: the object to be saved.
        """
        self.__logger.debug("Save the Pickle file and compress.")

        try:
            with bz2.BZ2File(self.__path, 'wb') as f:
                pickle.dump(objects, f, protocol=4)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: \n" + self.__path + "\n")
            print(e)
            sys.exit()
        except Exception as e:
            self.__logger.error(__name__ + " - Unable to save data to: \n" + self.__path + "\n")
            print(e)
            sys.exit()

        file_stats = os.stat(self.__path)
        self.__logger.info("Pickle Size: " + str(file_stats.st_size) + ".")

    def load(self) -> Any:
        """Load a serialised object, that was not compressed.
        :return: the loaded python object.
        """
        self.__logger.debug("Load a Pickle file.")
        try:
            with open(self.__path, 'rb') as f:
                objects = pickle.load(f)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: \n" + self.__path + "\n")
            print(e)
            sys.exit()
        except Exception as e:
            self.__logger.error(__name__ + " - Unable to load data from: \n" + self.__path + "\n")
            print(e)
            sys.exit()
        return objects

    def load_bz2(self) -> Any:
        """Load a serialised object, that was compressed (BZ2 compression).
        :return: the loaded python object.
        """
        self.__logger.debug("Save a compressed Pickle file.")
        try:
            with bz2.BZ2File(self.__path, 'rb') as f:
                objects = pickle.load(f)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: \n" + self.__path + "\n")
            print(e)
            sys.exit()
        except Exception as e:
            self.__logger.error(__name__ + " - Unable to load data from \n" + self.__path + "\n")
            print(e)
            sys.exit()
        return objects

    def size(self) -> OsStatResult:
        """Check the size of the saved file.
        :return: showing stat information of the file.
        """
        return os.stat(self.__path)
