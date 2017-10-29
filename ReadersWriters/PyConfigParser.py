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
""" It is an interface for reading and writing configuration file using 'configParser'.
"""

from typing import Dict, List
import sys
import configparser
import os
import logging

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class PyConfigParser:
    def __init__(self,
                 path: str,
                 app_name: str,
                 ext: str= ""):
        """Initialise the objects and constants.
        :param path: the file path of the configuration file.
        :param app_name: the application name, which will be used as the log file name.
        :param ext: the extension of the configuration file (default: 'ini').
        """
        self.__logger = logging.getLogger(app_name)
        self.__logger.debug(__name__)
        self.__path = os.path.abspath(path + "." + ext)
        self.__config = None
        self.refresh()

    def refresh(self):
        """Refresh the configuration file reader and Reset the constants.
        """
        self.__logger.debug("Refresh the Configuration file reader.")
        try:
            if not (os.path.exists(self.__path) and os.path.isfile(self.__path)):
                self.__logger.warning(__name__ + " - Configuration file does not exist: \n" + self.__path)
                self.reset()
            else:
                self.__config = configparser.ConfigParser()
                self.__config.optionxform = str  # make option case-sensitive
                self.__config.read(self.__path)
        except():
            self.__logger.error(__name__ + " - Error while opening a file: \n" + self.__path)
            sys.exit()

    def reset(self):
        """Reset the configuration file reader.
        """
        self.__logger.debug("Reset the Configuration file.")
        try:
            open(self.__path, 'w').close()
        except():
            self.__logger.error(__name__ + " - Could not create the config file: \n" + self.__path)
            sys.exit()

    def sections(self) -> List:
        """Get sections in the configuration file.
        :return: section names.
        """
        self.__logger.debug("Get Sections of the Configuration file.")
        return self.__config.sections()

    def subsections(self,
                    section: str) -> List:
        """Get sub-sections under the specified section name.
        :param section: the section name.
        :return: the sub-section names
        """
        self.__logger.debug("Get Subsections of the Configuration file.")
        return self.__config.items(section)

    def option(self,
               section: str,
               key: str) -> str:
        """Get the option for the specified key and section.
        :param section: a section in the configuration.
        :param key: a key of a section in the configuration.
        :return: the value (option) of the key.
        """
        self.__logger.debug("Read an Option in the Configuration file.")
        try:
            value = self.__config.get(section, key)
        except configparser.NoSectionError:
            self.__logger.error(__name__ + " - Invalid Section: [Section: " +
                                str(section) + "; Key:" + str(key) + "]")
            sys.exit()
        except configparser.NoOptionError:
            self.__logger.error(__name__ + " - Invalid Option: [Section: " +
                                str(section) + "; Key:" + str(key) + "]")
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Invalid Configuration: [Section: " +
                                str(section) + "; Key:" + str(key) + "]")
            sys.exit()
        return value

    def options(self,
                section: str,
                keys: List=None) -> Dict:
        """Get the options for all or the specified keys in the section.
        :param section: a section in the configuration.
        :param keys: keys of a section (option) in the configuration.
        :return: the values (options) of the keys.
        """
        self.__logger.debug("Read Section Options in the Configuration file.")
        values = []
        try:
            if keys is None:
                values = dict(self.subsections(section))
            else:
                for k in keys:
                    values.append(self.__config.get(section, k))
        except configparser.NoSectionError:
            self.__logger.error(__name__ + " - Invalid Section: " +
                                "[Section: " + str(section) + "]")
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Invalid Configuration.")
            sys.exit()
        return values

    def read_dict(self,
                  section: str=None) -> Dict:
        """Read the configuration and save into dictionary.
        :param section: the section name
        :return: the read configuration file
        """
        self.__logger.debug("Read into Dictionary.")
        dic = dict()
        sections = self.sections()

        if section is None:
            for section in sections:
                dic[section] = self.options(section)
                for k, v in dic[section].items():
                    dic[section][k] = str(v).split(',')
        else:
            if section in set(sections):
                dic = self.options(section)
                for k, v in dic.items():
                    dic[k] = str(v).split(',')
        return dic

    def write_dict(self,
                   dic: Dict,
                   section: str,
                   append: str=False):
        """Write from the inputted dictionary into the configuration file.
        :param dic: the inputted dictionary to write into the configuration file.
        :param section: the section name.
        :param append: indicates if the write appends to any existing configuration file.
        """
        self.__logger.debug("Write from Dictionary.")
        # set
        config = configparser.RawConfigParser()
        config.optionxform = str  # make option is case-sensitive
        config.add_section(section)
        keys = list(dic.keys())

        for key in keys:
            if isinstance(dic[key], list):
                config.set(section, key, ','.join(dic[key]))
            else:
                config.set(section, key, dic[key])

        # write
        if append is False:
            with open(self.__path, 'w') as file:
                config.write(file)
        else:
            with open(self.__path, 'a') as file:
                config.write(file)

    def write_option(self,
                     section: str,
                     option: str,
                     value: str):
        """Remove then add the specified option to the configuration file.
        :param section: the section name.
        :param option: the option name to be removed then added with new value.
        :param value: the option value to be removed then added with new value.
        """
        self.__logger.debug("Write an Option into the Configuration file.")
        self.__remove_option(section, option)
        self.__add_option(section, option, value)

    def __remove_option(self,
                        section: str,
                        option: str):
        """Remove an option from the configuration file.
        :param section: the section name.
        :param option: the option name to remove.
        """
        self.__logger.debug("Remove an Option from the Configuration file.")
        section = "[" + section + "]"
        option += "="
        match = False

        # read
        with open(self.__path, 'r') as file:
            lines = file.readlines()

        # delete
        with open(self.__path, 'w') as file:
            # remove
            for line in lines:
                if line.strip().startswith(section):
                    match = True
                if match is True and line.replace(' ', '').startswith(option):
                    match = False
                    continue
                file.write(line)

    def __add_option(self,
                     section: str,
                     option: str,
                     value: str):
        """Add an option to the configuration file.
        :param section: the section name.
        :param option: the option name to be written.
        :param value: the option value to be written.
        """
        self.__logger.debug("Add an Option to the Configuration file.")
        section = "[" + section + "]"
        match = False

        # read
        with open(self.__path, 'r') as file:
            lines = file.readlines()

        # append
        with open(self.__path, 'w') as file:
            for line in lines:
                if line.strip().startswith(section):
                    file.write(line)
                    file.write(option + " = " + ",".join(value) + "\n")
                    match = True
                else:
                    file.write(line)
            if match is False:
                file.write("\n" + section + "\n" + option + " = " + ",".join(value) + "\n")
