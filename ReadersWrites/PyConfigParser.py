#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import configparser
import os
import logging

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class PyConfigParser:

    def __init__(self, path, app_name, extension=""):
        self.__logger = logging.getLogger(app_name)
        self.__logger.debug(__name__)
        self.__path = os.path.abspath(path + extension)
        self.__config = None
        self.refresh()

    def refresh(self):
        try:
            if not (os.path.exists(self.__path) and os.path.isfile(self.__path)):
                self.__logger.warning(__name__ + " - Configuration file does not exist:  " + self.__path)
                self.reset()
            else:
                self.__config = configparser.ConfigParser()
                self.__config.optionxform = str  # make option case-sensitive
                self.__config.read(self.__path)
        except():
            self.__logger.error(__name__ + " - Error while opening a file: " + self.__path)
            sys.exit()

    def reset(self):
        self.__logger.debug(__name__)
        try:
            open(self.__path, 'w').close()
        except():
            self.__logger.error(__name__ + " - Could not create the config file: " + self.__path)
            sys.exit()

    def sections(self):
        """
        Get sections
        :return: return section names
        """
        return self.__config.sections()

    def subsections(self, section):
        return self.__config.items(section)

    def option(self, section, key):
        """
        Get the option for the specified key in the section
        :param section: A section in the configuration
        :param key: A key of a section in the configuration
        :return: The value (option) of the key
        """
        try:
            value = self.__config.get(section, key)
        except configparser.NoSectionError:
            self.__logger.error(__name__ + " - Invalid Section: " + "Section: " + section + "; Key:" + key)
            sys.exit()
        except configparser.NoOptionError:
            self.__logger.error(__name__ + " - Invalid Option: " + "Section: " + section + "; Key:" + key)
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Invalid Configuration: " + "Section: " + section + "; Key:" + key)
            sys.exit()
        return value

    def options(self, section, keys=None):
        """
        Get the options for all or the specified keys in the section
        :param section: A section in the configuration
        :param keys: Keys of a section (option) in the configuration
        :return: The values (options) of the keys
        """
        self.__logger.debug(__name__)
        values = []
        try:
            if keys is None:
                values = dict(self.subsections(section))
            else:
                for k in keys:
                    values.append(self.__config.get(section, k))
        except configparser.NoSectionError:
            self.__logger.error(__name__ + " - Invalid Section: " +
                                "Section: " + section)
            sys.exit()
        except():
            self.__logger.error(__name__ + " - Invalid Configuration! ")
            sys.exit()
        return values

    def read_dict(self, section=None):
        self.__logger.debug(__name__)
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

    def write_dict(self, dic, section, key_order=None, append=False):
        self.__logger.debug(__name__)
        # set
        config = configparser.RawConfigParser()
        config.optionxform = str  # make option case-sensitive
        config.add_section(section)
        if key_order is None:
            keys = list(dic.keys())
        else:
            keys = key_order

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

    def write_option(self, section, option, value):
        self.__logger.debug(__name__)
        self.__remove_option(section, option)
        self.__add_option(section, option, value)

    def __remove_option(self, section, option):
        self.__logger.debug(__name__)
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

    def __add_option(self, section, option, value):
        self.__logger.debug(__name__)
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
