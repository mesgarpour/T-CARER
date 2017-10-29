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
""" It reads and process variables.
"""

from typing import List, TypeVar, Dict, Callable
import sys
import logging
import pandas as pd
from ReadersWriters.ReadersWriters import ReadersWriters
from Features.FeatureParser import FeatureParser
from Configs.CONSTANTS import CONSTANTS

PandasDataFrame = TypeVar('DataFrame')
FeaturesFeatureParser = TypeVar('FeatureParser')

__author__ = "Mohsen Mesgarpour"
__copyright__ = "Copyright 2016, https://github.com/mesgarpour"
__credits__ = ["Mohsen Mesgarpour"]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Mohsen Mesgarpour"
__email__ = "mohsen.mesgarpour@gmail.com"
__status__ = "Release"


class Variables:
    def __init__(self,
                 model_features_table: str,
                 input_path: str,
                 output_path: str,
                 input_features_configs: str,
                 output_table: str):
        """Initialise the objects and constants.
        :param model_features_table: the feature table name.
        :param input_path: the input path.
        :param output_path: the output path.
        :param input_features_configs: the input features' configuration file.
        :param output_table: the output table name.
        """
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

    def set(self,
            input_schemas: List,
            input_tables: List,
            history_tables: List,
            column_index: str,
            query_batch_size: int):
        """Set the variables by reading the selected features from MySQL database.
        :param input_schemas: the mysql database schemas.
        :param input_tables: the mysql table names.
        :param history_tables: the source tables' alias names (a.k.a. history table name) that features belong to
            (e.g. inpatient, or outpatient).
        :param column_index: the name of index column (unique integer value) in the database table, which is used
            for batch reading the input.
        :param query_batch_size: the number of rows to be read in each batch.
        :return:
        """
        self.__logger.debug(__name__)
        query_batch_start, query_batch_max = self.__init_batch(input_schemas[0], input_tables[0])
        features_names, features_dtypes = self.__set_features_names_types()
        self.__validate_mysql_names(input_schemas, input_tables)
        prevalence = self.__init_prevalence(input_schemas, input_tables, history_tables)
        self.__set_batch(features_names, features_dtypes, input_schemas, input_tables, history_tables, column_index,
                         prevalence, query_batch_start, query_batch_max, query_batch_size)

    def __init_settings(self,
                        input_path: str,
                        input_features_configs: str) -> PandasDataFrame:
        """Read and set the settings of input variables that are selected.
        :param input_path: the path of the input file.
        :param input_features_configs: the input features' configuration file.
        :return: the input variables settings.
        """
        self.__logger.debug(__name__)
        variables_settings = self.__readers_writers.load_csv(input_path, input_features_configs, 0, True)
        variables_settings = variables_settings.loc[
            (variables_settings["Selected"] == 1) &
            (variables_settings["Table_Reference_Name"] == self.__model_features_table)]
        variables_settings = variables_settings.reset_index()
        return variables_settings

    def __init_features_names(self) -> Dict:
        """Generate the features names, based on variable name, source table alias name (a.k.a. history table
            name), and the aggregation function name.
        :return: the name of features.
        """
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

    def __init_features_dtypes(self) -> Dict:
        """Generate the features types, based on the input configuration file.
        :return: the dtypes of features.
        """
        self.__logger.debug(__name__)
        table_history_names = set(self.__variables_settings["Table_History_Name"])
        features_dtypes = dict(zip(table_history_names, [[] for _ in range(len(table_history_names))]))
        for _, row in self.__variables_settings.iterrows():
            feature_types = row["Variable_dType"].replace(' ', '').split(',')
            for feature_type in feature_types:
                features_dtypes[row["Table_History_Name"]].append(feature_type)
        return features_dtypes

    def __init_output(self,
                      output_path: str,
                      output_table: str):
        """Initialise the output file by writing the header row.
        :param output_path: the output path.
        :param output_table: the output table name.
        """
        self.__logger.debug(__name__)
        keys = sorted(self.__features_dic_names.keys())
        features_names = [f for k in keys for f in self.__features_dic_names[k]]
        self.__readers_writers.reset_csv(output_path, output_table)
        self.__readers_writers.save_csv(output_path, output_table, features_names, append=False)

    def __init_prevalence(self,
                          input_schemas: List,
                          input_tables: List,
                          history_tables: List)-> Dict:
        """Generate the prevalence dictionary of values for all the variables.
        :param input_schemas: the mysql database schemas.
        :param input_tables: the mysql table names.
        :param history_tables: the source tables' alias names (a.k.a. history table name) that features belong to
            (e.g. inpatient, or outpatient).
        :return: the prevalence dictionary of values for all the variables.
        """
        self.__readers_writers.save_text(
            self.__output_path, self.__output_table,
            ["Feature Name", "Top Prevalence Feature Name"], append=False, ext="ini")
        self.__readers_writers.save_text(
            self.__output_path, self.__output_table,
            ["Feature Name", "Prevalence & Freq."], append=False, ext="txt")
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
                                                             append=True, ext="ini")
        return prevalence

    def __init_prevalence_read(self,
                               input_schema: str,
                               input_table: str,
                               variable_name: str) -> PandasDataFrame:
        """Read a variable from database, to calculate the prevalence of the values.
        :param input_schema: the mysql database schema.
        :param input_table: the mysql database table.
        :param variable_name: the variable name.
        :return: the selected variable.
        """
        query = "SELECT `" + variable_name + "` FROM `" + input_table + "`;"
        return self.__readers_writers.load_mysql_query(query, input_schema, dataframing=True)

    def __init_batch(self,
                     input_schema: str,
                     input_table: str) -> [int, int]:
        """Find the minimum and maximum value of the index column, to use when reading mysql tables in
            batches.
        :param input_schema: the mysql database schema.
        :param input_table: the mysql database table.
        :return: the minimum and maximum of the index column.
        """
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
        """Produce the sorted lists of features names and features dtypes.
        :return: the sorted lists of features names and features dtypes.
        """
        self.__logger.debug(__name__)
        keys = sorted(self.__features_dic_names.keys())
        features_names = [f for k in keys for f in self.__features_dic_names[k]]
        features_dtypes = [pd.Series(dtype=f) for k in keys for f in self.__features_dic_dtypes[k]]
        features_dtypes = pd.DataFrame(dict(zip(features_names, features_dtypes))).dtypes
        return features_names, features_dtypes

    def __set_batch(self,
                    features_names: list,
                    features_dtypes: Dict,
                    input_schemas: List,
                    input_tables: List,
                    history_tables: List,
                    column_index: str,
                    prevalence: Dict,
                    query_batch_start: int,
                    query_batch_max: int,
                    query_batch_size: int):
        """Using batch processing first read variables, then generate features and write them into output.
        :param features_names: the name of features that are selected.
        :param features_dtypes: the dtypes of features that are selected.
        :param input_schemas: the mysql database schemas.
        :param input_tables: the mysql table names.
        :param history_tables: the source tables' alias names (a.k.a. history table name) that features belong to
            (e.g. inpatient, or outpatient).
        :param column_index: the name of index column (unique integer value) in the database table, which is used
            for batch reading the input.
        :param prevalence: the prevalence dictionary of values for all the variables.
        :param query_batch_start: the minimum value of the column index.
        :param query_batch_max: the maximum value of the column index.
        :param query_batch_size: the number of rows to be read in each batch.
        """
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

    def __set_batch_read(self,
                         input_schema: str,
                         input_table: str,
                         step: int,
                         column_index: str,
                         query_batch_start: int,
                         query_batch_max: int,
                         query_batch_size: int) -> Callable[[PandasDataFrame, None], None]:
        """Read the queried variables.
        :param input_schema: the mysql database schema.
        :param input_table: the mysql database table.
        :param step: the batch id.
        :param column_index: the name of index column (unique integer value) in the database table, which is used
            for batch reading the input.
        :param query_batch_start: the minimum value of the column index.
        :param query_batch_max: the maximum value of the column index.
        :param query_batch_size: the number of rows to be read in each batch.
        :return: the queried variables.
        """
        step_start = query_batch_start + step * query_batch_size
        step_end = step_start + query_batch_size
        if step_start >= query_batch_max:
            return None
        # read
        query = "SELECT * FROM `" + input_table + \
                "` WHERE `" + str(column_index) + "` >= " + str(step_start) + \
                " AND `" + str(column_index) + "` < " + str(step_end) + ";"
        return self.__readers_writers.load_mysql_query(query, input_schema, dataframing=True)

    def __set_batch_process(self,
                            feature_parser: FeaturesFeatureParser,
                            history_table: str,
                            features: PandasDataFrame,
                            variables: PandasDataFrame,
                            prevalence: List) -> PandasDataFrame:
        """Process variables and generate features.
        :param feature_parser:
        :param history_table: the source table alias name (a.k.a. history table name) that features belong to
            (e.g. inpatient, or outpatient).
        :param features: the output features.
        :param variables: the input variables.
        :param prevalence: the prevalence dictionary of values for all the variables.
        :return: the generated features.
        """
        return feature_parser.generate(history_table, features, variables, prevalence)

    def __set_batch_write(self,
                          features: PandasDataFrame):
        """Write the features into an output file.
        :param features: the output features.
        """
        self.__readers_writers.save_csv(self.__output_path, self.__output_table, features, append=True)

    def __validate_mysql_names(self,
                               input_schemas: List,
                               history_tables: List):
        """Validate mysql tables and their columns, and generate exception if table/column name is invalid.
        :param input_schemas: the mysql database schemas.
        :param history_tables: the source tables' alias names (a.k.a. history table name) that features belong to
            (e.g. inpatient, or outpatient).
        """
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
