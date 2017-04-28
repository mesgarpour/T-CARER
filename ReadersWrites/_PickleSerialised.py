#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pickle
import bz2
import sys
import os
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


class PickleSerialised:

    def __init__(self):
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__path = None

    def set(self, path, title, ext):
        self.__logger.debug(__name__)
        print( path)
        self.__path = os.path.join(path, title + "." + ext)
        print(self.__path)

    def exists(self):
        self.__logger.debug(__name__)
        return os.path.isfile(self.__path)

    def save(self, objects):
        self.__logger.debug(__name__)
        self.__logger.info("Saving Pickled file")

        try:
            with open(self.__path, 'wb') as f:
                pickle.dump(objects, f, protocol=4)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n")
            print(e)
            sys.exit()
        except Exception as e:
            self.__logger.error(__name__ + " - Unable to save data to " + self.__path + "\n")
            print(e)
            sys.exit()

        file_stats = os.stat(self.__path)
        self.__logger.info("Pickle size: " + str(file_stats.st_size))

    def save_bz2(self, objects):
        self.__logger.debug(__name__)
        self.__logger.info("Saving Pickled file")

        try:
            with bz2.BZ2File(self.__path, 'wb') as f:
                pickle.dump(objects, f, protocol=4)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n")
            print(e)
            sys.exit()
        except Exception as e:
            self.__logger.error(__name__ + " - Unable to save data to " + self.__path + "\n")
            print(e)
            sys.exit()

        file_stats = os.stat(self.__path)
        self.__logger.info("Pickle size: " + str(file_stats.st_size))

    def load(self):
        self.__logger.debug(__name__)
        self.__logger.info("Loading Pickled file")
        try:
            with open(self.__path, 'rb') as f:
                objects = pickle.load(f)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n")
            print(e)
            sys.exit()
        except Exception as e:
            self.__logger.error(__name__ + " - Unable to load data from " + self.__path + "\n")
            print(e)
            sys.exit()
        return objects

    def load_bz2(self):
        self.__logger.debug(__name__)
        self.__logger.info("Loading Pickled file")
        try:
            with bz2.BZ2File(self.__path, 'rb') as f:
                objects = pickle.load(f)
        except (OSError, IOError) as e:
            self.__logger.error(__name__ + " - Can not open the file: " + self.__path + "\n")
            print(e)
            sys.exit()
        except Exception as e:
            self.__logger.error(__name__ + " - Unable to load data from " + self.__path + "\n")
            print(e)
            sys.exit()
        return objects

    def size(self):
        return os.stat(self.__path)
