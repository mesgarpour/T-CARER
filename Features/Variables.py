#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
import logging
import pandas as pd
from ReadersWrites.ReadersWriters import ReadersWriters
from Features.FeatureParser import FeatureParser
from Configs.CONSTANTS import CONSTANTS

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.x"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Development"


class Variables:
    def __init__(self, model_features_table, input_path, output_path, input_features_configs, output_table):
        self.__logger = logging.getLogger(CONSTANTS.app_name)
        self.__logger.debug(__name__)
        self.__model_features_table = model_features_table
        self.__output_path = output_path
        self.__output_table = output_table
        self.__readers_writers = ReadersWriters()
        # initialise settings
        self.__variables_settings = self.__init_settings(input_path, input_features_configs)
        self.__features_dic_names = self.__init_features_names()
        self.__features_dic_dtypes = self.__init_features_dtypes()
        self.__init_output(output_path, output_table)

    def set(self, input_schemas, input_tables, history_tables, column_index, query_batch_size):
        self.__logger.debug(__name__)
        query_batch_start, query_batch_max = self.__init_batch(input_schemas[0], input_tables[0])
        features_names, features_dtypes = self.__set_features_names_types()
        self.__validate_mysql_names(input_schemas, input_tables)
        prevalence = self.__init_prevalence(input_schemas, input_tables, history_tables)
        self.__set_batch(features_names, features_dtypes, input_schemas, input_tables, history_tables, column_index,
                         prevalence, query_batch_start, query_batch_max, query_batch_size)

    def __init_settings(self, input_path, input_features_configs):
        self.__logger.debug(__name__)
        variables_settings = self.__readers_writers.load_csv(input_path, input_features_configs, 0, True)
        variables_settings = variables_settings.loc[
            (variables_settings["Selected"] == 1) &
            (variables_settings["Table_Reference_Name"] == self.__model_features_table)]
        variables_settings = variables_settings.reset_index()
        return variables_settings

    def __init_features_names(self):
        self.__logger.debug(__name__)
        table_history_names = set(self.__variables_settings["Table_History_Name"])
        features_names = dict(zip(table_history_names, [[] for _ in range(len(table_history_names))]))
        for _, row in self.__variables_settings.iterrows():
            if not pd.isnull(row["Variable_Aggregation"]):
                postfixes = row["Variable_Aggregation"].replace(' ', '').split(',')
                for postfix in postfixes:
                    features_names[row["Table_History_Name"]].append(row["Variable_Name"] + "_" + postfix)
            else:
                features_names[row["Table_History_Name"]].append(row["Variable_Name"])
        return features_names

    def __init_features_dtypes(self):
        self.__logger.debug(__name__)
        table_history_names = set(self.__variables_settings["Table_History_Name"])
        features_dtypes = dict(zip(table_history_names, [[] for _ in range(len(table_history_names))]))
        for _, row in self.__variables_settings.iterrows():
            feature_types = row["Variable_dType"].replace(' ', '').split(',')
            for feature_type in feature_types:
                features_dtypes[row["Table_History_Name"]].append(feature_type)
        return features_dtypes

    def __init_output(self, output_path, output_table):
        self.__logger.debug(__name__)
        keys = sorted(self.__features_dic_names.keys())
        features_names = [f for k in keys for f in self.__features_dic_names[k]]
        self.__readers_writers.reset_csv(output_path, output_table)
        self.__readers_writers.save_csv(output_path, output_table, features_names, append=False)

    def __init_prevalence(self, input_schemas, input_tables, history_tables):
        self.__readers_writers.save_text(
            self.__output_path, self.__output_table,
            ["Feature Name", "Top Prevalence Feature Name"], append=False, extension="ini")
        self.__readers_writers.save_text(
            self.__output_path, self.__output_table,
            ["Feature Name", "Prevalence & Freq."], append=False, extension="txt")
        feature_parser = FeatureParser(self.__variables_settings, self.__output_path, self.__output_table)
        prevalence = dict()

        # for tables
        for table_i in range(len(input_schemas)):
            variables_settings = self.__variables_settings[
                self.__variables_settings["Table_History_Name"] == history_tables[table_i]]
            prevalence[input_tables[table_i]] = dict()

            # for features
            for _, row in variables_settings.iterrows():
                self.__logger.info("Prevalence: " + row["Variable_Name"] + " ...")
                if not pd.isnull(row["Variable_Aggregation"]):
                    # read features
                    variables = self.__init_prevalence_read(
                        input_schemas[table_i], input_tables[table_i], row["Variable_Name"])

                    # validate
                    if variables is None or len(variables) == 0:
                        continue

                    # prevalence
                    prevalence[input_tables[table_i]][row["Variable_Name"]] = \
                        feature_parser.prevalence(variables[row["Variable_Name"]], row["Variable_Name"])

                    # for sub features
                    postfixes = row["Variable_Aggregation"].replace(' ', '').split(',')
                    for p in range(len(postfixes)):
                        feature_name = row["Variable_Name"] + "_" + postfixes[p]
                        if len(postfixes[p]) > 11 and postfixes[p][0:11] == "prevalence_":
                            index = int(postfixes[p].split('_')[1]) - 1
                            feature_name_prevalence = "None"
                            if index < len(prevalence[input_tables[table_i]][row["Variable_Name"]]):
                                feature_name_prevalence = \
                                    feature_name + "_" + \
                                    str(prevalence[input_tables[table_i]][row["Variable_Name"]][index])
                            # save prevalence
                            self.__readers_writers.save_text(self.__output_path, self.__output_table,
                                                             [feature_name, feature_name_prevalence],
                                                             append=True, extension="ini")
        return prevalence

    def __init_prevalence_read(self, input_schema, input_table, variable_name):
        query = "SELECT `" + variable_name + "` FROM `" + input_table + "`;"
        return self.__readers_writers.load_mysql_query(query, input_schema, dataframing=True)

    def __init_batch(self, input_schema, input_table):
        self.__logger.debug(__name__)
        query = "select min(localID), max(localID) from `" + input_table + "`;"
        output = list(self.__readers_writers.load_mysql_query(query, input_schema, dataframing=False))
        if [r[0] for r in output][0] is None:
            self.__logger.error(__name__ + " No data is found: " + query)
            sys.exit()

        query_batch_start = int([r[0] for r in output][0])
        query_batch_max = int([r[1] for r in output][0])
        return query_batch_start, query_batch_max

    def __set_features_names_types(self):
        self.__logger.debug(__name__)
        keys = sorted(self.__features_dic_names.keys())
        features_names = [f for k in keys for f in self.__features_dic_names[k]]
        features_dtypes = [pd.Series(dtype=f) for k in keys for f in self.__features_dic_dtypes[k]]
        features_dtypes = pd.DataFrame(dict(zip(features_names, features_dtypes))).dtypes
        return features_names, features_dtypes

    def __set_batch(self, features_names, features_dtypes, input_schemas, input_tables, history_tables, column_index,
                    prevalence, query_batch_start, query_batch_max, query_batch_size):
        self.__logger.debug(__name__)
        feature_parser = FeatureParser(self.__variables_settings, self.__output_path, self.__output_table)
        step = -1
        batch_break = False

        while not batch_break:
            step += 1
            features = None
            for table_i in range(len(input_schemas)):
                self.__logger.info("Batch: " + str(step) + "; Table: " + input_tables[table_i])

                # read job
                variables = self.__set_batch_read(input_schemas[table_i], input_tables[table_i], step, column_index,
                                                  query_batch_start, query_batch_max, query_batch_size)

                # validate
                if variables is None:
                    batch_break = True
                    break
                elif len(variables) == 0:
                    continue

                # process job
                if features is None:
                    features = pd.DataFrame(0, index=range(len(variables)), columns=features_names)
                    features = features.astype(dtype=features_dtypes)
                features = self.__set_batch_process(
                    feature_parser, history_tables[table_i], features, variables, prevalence[input_tables[table_i]])

            # write job
            if features is not None:
                features = features.astype(dtype=features_dtypes)
                self.__set_batch_write(features)

    def __set_batch_read(self, input_schema, input_table, step, column_index,
                         query_batch_start, query_batch_max, query_batch_size):
        step_start = query_batch_start + step * query_batch_size
        step_end = step_start + query_batch_size
        if step_start >= query_batch_max:
            return None
        # read
        query = "SELECT * FROM `" + input_table + \
                "` WHERE `" + column_index + "` >= " + str(step_start) + \
                " AND `" + column_index + "` < " + str(step_end) + ";"
        return self.__readers_writers.load_mysql_query(query, input_schema, dataframing=True)

    def __set_batch_process(self, feature_parser, history_table, features, variables, prevalence):
        return feature_parser.generate(history_table, features, variables, prevalence)

    def __set_batch_write(self, features):
        self.__readers_writers.save_csv(self.__output_path, self.__output_table, features, append=True)

    def __validate_mysql_names(self, input_schemas, history_tables):
        # for tables
        for table_i in range(len(input_schemas)):
            variables_settings = self.__variables_settings[
                self.__variables_settings["Table_History_Name"] == history_tables[table_i]]
            # validate table name
            if not self.__readers_writers.exists_mysql(
                    input_schemas[table_i], history_tables[table_i]):
                self.__logger.error(__name__ + " - Table does not exist: " + history_tables[table_i])
                sys.exit()

            # for features
            for _, row in variables_settings.iterrows():
                # validate column name
                if not self.__readers_writers.exists_mysql_column(
                        input_schemas[table_i], history_tables[table_i], row["Variable_Name"]):
                    self.__logger.error(__name__ + " - Column does not exist: " + row["Variable_Name"])
                    sys.exit()
