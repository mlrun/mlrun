# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import pandas as pd

from ..feature_vector import OfflineVectorResponse
from .base import BaseMerger


class LocalFeatureMerger(BaseMerger):
    engine = "local"

    def __init__(self, vector, **engine_args):
        super().__init__(vector, **engine_args)

    def _generate_vector(
        self,
        entity_rows,
        entity_timestamp_column,
        feature_set_objects,
        feature_set_fields,
        start_time=None,
        end_time=None,
        query=None,
    ):

        feature_sets = []
        dfs = []
        keys = (
            []
        )  # the struct of key is [[[],[]], ..] So that each record indicates which way the corresponding
        # featureset is connected to the previous one, and within each record the left keys are indicated in index 0
        # and the right keys in index 1, this keys will be the keys that will be used in this join
        all_columns = []

        fs_link_list = self._create_linked_relation_list(
            feature_set_objects, feature_set_fields
        )

        for node in fs_link_list:
            name = node.name
            feature_set = feature_set_objects[name]
            feature_sets.append(feature_set)
            columns = feature_set_fields[name]
            column_names = [name for name, alias in columns]

            for col in node.data["save_cols"]:
                if col not in column_names:
                    self._append_drop_column(col)
            column_names += node.data["save_cols"]

            # handling case where there are multiple feature sets and user creates vector where entity_timestamp_
            # column is from a specific feature set (can't be entity timestamp)
            if (
                entity_timestamp_column in column_names
                or feature_set.spec.timestamp_key == entity_timestamp_column
            ):
                df = feature_set.to_dataframe(
                    columns=column_names,
                    start_time=start_time,
                    end_time=end_time,
                    time_column=entity_timestamp_column,
                )
            else:
                df = feature_set.to_dataframe(
                    columns=column_names,
                    time_column=entity_timestamp_column,
                )
            if df.index.names[0]:
                df.reset_index(inplace=True)
            column_names += node.data["save_index"]
            node.data["save_cols"] += node.data["save_index"]
            entity_timestamp_column_list = (
                [entity_timestamp_column]
                if entity_timestamp_column
                else feature_set.spec.timestamp_key
            )
            if entity_timestamp_column_list:
                column_names += entity_timestamp_column_list
                node.data["save_cols"] += entity_timestamp_column_list
            # rename columns to be unique for each feature set
            rename_col_dict = {
                col: f"{col}_{name}"
                for col in column_names
                if col not in node.data["save_cols"]
            }
            df.rename(
                columns=rename_col_dict,
                inplace=True,
            )

            dfs.append(df)
            keys.append([node.data["left_keys"], node.data["right_keys"]])

            # update alias according to the unique column name
            new_columns = []
            for col, alias in columns:
                if col in rename_col_dict and alias:
                    new_columns.append((rename_col_dict[col], alias))
                elif col in rename_col_dict and not alias:
                    new_columns.append((rename_col_dict[col], col))
                else:
                    new_columns.append((col, alias))
            all_columns.append(new_columns)
            self._update_alias(
                dictionary={name: alias for name, alias in new_columns if alias}
            )

        self.merge(
            entity_df=entity_rows,
            entity_timestamp_column=entity_timestamp_column,
            featuresets=feature_sets,
            featureset_dfs=dfs,
            keys=keys,
            all_columns=all_columns,
        )

        self._result_df.drop(columns=self._drop_columns, inplace=True, errors="ignore")

        # renaming all columns according to self._alias
        self._result_df.rename(
            columns=self._alias,
            inplace=True,
        )
        if self.vector.status.label_column:
            self._result_df.dropna(
                subset=[self.vector.status.label_column],
                inplace=True,
            )
        # filter joined data frame by the query param
        if query:
            self._result_df.query(query, inplace=True)

        if self._drop_indexes:
            self._result_df.reset_index(drop=True, inplace=True)
        else:
            self._set_indexes(self._result_df)

        self._write_to_target()

        return OfflineVectorResponse(self)

    def _asof_join(
        self,
        entity_df,
        entity_timestamp_column: str,
        featureset,
        featureset_df,
        left_keys: list,
        right_keys: list,
        columns: list,
    ):

        indexes = None
        if not right_keys:
            indexes = list(featureset.spec.entities.keys())
        index_col_not_in_entity = "index" not in entity_df.columns
        index_col_not_in_featureset = "index" not in featureset_df.columns
        entity_df[entity_timestamp_column] = pd.to_datetime(
            entity_df[entity_timestamp_column]
        )
        featureset_df[featureset.spec.timestamp_key] = pd.to_datetime(
            featureset_df[featureset.spec.timestamp_key]
        )
        entity_df.sort_values(by=entity_timestamp_column, inplace=True)
        featureset_df.sort_values(by=entity_timestamp_column, inplace=True)

        merged_df = pd.merge_asof(
            entity_df,
            featureset_df,
            left_on=entity_timestamp_column,
            right_on=featureset.spec.timestamp_key,
            by=indexes,
            left_by=left_keys or None,
            right_by=right_keys or None,
            suffixes=("", f"_{featureset.metadata.name}_"),
        )
        for col in merged_df.columns:
            if re.findall(f"_{featureset.metadata.name}_$", col):
                self._append_drop_column(col)

        # Undo indexing tricks for asof merge
        # to return the correct indexes and not
        # overload `index` columns
        if (
            indexes
            and "index" not in indexes
            and index_col_not_in_entity
            and index_col_not_in_featureset
            and "index" in merged_df.columns
        ):
            merged_df.drop(columns="index", inplace=True)
        return merged_df

    def _join(
        self,
        entity_df,
        entity_timestamp_column: str,
        featureset,
        featureset_df,
        left_keys: list,
        right_keys: list,
        columns: list,
    ):
        fs_name = featureset.metadata.name
        merged_df = pd.merge(
            entity_df,
            featureset_df,
            how=self._join_type,
            left_on=left_keys,
            right_on=right_keys,
            suffixes=("", f"_{fs_name}_"),
        )
        for col in merged_df.columns:
            if re.findall(f"_{fs_name}_$", col):
                self._append_drop_column(col)
        return merged_df
