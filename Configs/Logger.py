import os
import logging

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class Logger:
    def __init__(self, path, app_name="T-CARER", extension="log"):
        # create logger
        logger = logging.getLogger(app_name)
        logger.setLevel(logging.DEBUG)
        path_full = os.path.abspath(os.path.join(path, app_name + "." + extension))

        # create file handler which logs even debug messages
        fh = logging.FileHandler(path_full, mode='w')
        fh.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # create formatter and add it to the handlers
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        # output log
        logger.info("creating " + path_full + " file")
