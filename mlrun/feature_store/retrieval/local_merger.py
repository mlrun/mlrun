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
        feature_sets_names = []
        keys = []
        all_columns = list()
        self._parse_relations(
            feature_set_objects=feature_set_objects,
            feature_set_fields=feature_set_fields,
        )

        # extract all dfs from the feature sets.
        for name, columns in feature_set_fields.items():
            feature_set = feature_set_objects[name]
            feature_sets.append(feature_set)
            column_names = [name for name, alias in columns]
            feature_sets_names.append(name)

            # updating the columns list that needed for tho join.
            # Build left and right keys for the join apply
            right_keys = []
            left_keys = []
            temp_key = []
            for key, dict_relation in self._relation.items():
                if re.findall(f":{name}$", key):
                    for left, right in dict_relation.items():
                        if (
                            right not in column_names
                            and right not in feature_set.spec.entities.keys()
                        ):
                            column_names.append(right)
                            if self._drop_indexes:
                                self._append_drop_column(right)
                        if key.split(":")[0] in feature_sets_names:
                            right_keys.append(right)
                            left_keys.append(left)
                        temp_key.append(right)
                elif re.findall(f"^{name}:", key):
                    for left, right in dict_relation.items():
                        if (
                            left not in column_names
                            and left not in feature_set.spec.entities.keys()
                        ):
                            column_names.append(left)
                            if self._drop_indexes:
                                self._append_drop_column(left)
                        if key.split(":")[1] in feature_sets_names:
                            right_keys.append(left)
                            left_keys.append(right)
                        temp_key.append(left)

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
            # rename columns to be unique for each feature set
            rename_col_dict = {
                col: f"{col}_{name}" for col in column_names if col not in temp_key
            }
            df.rename(
                columns=rename_col_dict,
                inplace=True,
            )

            dfs.append(df)
            keys.append([left_keys, right_keys])

            # update alias according to the unique column name
            new_columns = []
            for col, alias in columns:
                if col in rename_col_dict and alias:
                    new_columns.append((rename_col_dict[col], alias))
                elif col in rename_col_dict and not alias:
                    new_columns.append((rename_col_dict[col], alias))
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
            self._result_df = self._result_df.dropna(
                subset=[self.vector.status.label_column]
            )
        # filter joined data frame by the query param
        if query:
            self._result_df.query(query, inplace=True)

        if self._drop_indexes:
            self._result_df.reset_index(drop=True, inplace=True)
        self._write_to_target()

        # check if need to set indices
        self._result_df = self._set_indexes(self._result_df)
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
        # Sort left and right keys
        if type(entity_df.index) != pd.RangeIndex:
            entity_df = entity_df.reset_index()
        if type(featureset_df.index) != pd.RangeIndex:
            featureset_df = featureset_df.reset_index()
        entity_df[entity_timestamp_column] = pd.to_datetime(
            entity_df[entity_timestamp_column]
        )
        featureset_df[featureset.spec.timestamp_key] = pd.to_datetime(
            featureset_df[featureset.spec.timestamp_key]
        )
        entity_df = entity_df.sort_values(by=entity_timestamp_column)
        featureset_df = featureset_df.sort_values(by=entity_timestamp_column)

        merged_df = pd.merge_asof(
            entity_df,
            featureset_df,
            left_on=entity_timestamp_column,
            right_on=featureset.spec.timestamp_key,
            by=indexes,
            left_by=left_keys,
            right_by=right_keys,
        )

        # Undo indexing tricks for asof merge
        # to return the correct indexes and not
        # overload `index` columns
        if (
            "index" not in indexes
            and index_col_not_in_entity
            and index_col_not_in_featureset
            and "index" in merged_df.columns
        ):
            merged_df = merged_df.drop(columns="index")
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
        indexes = None
        if not right_keys:
            indexes = list(featureset.spec.entities.keys())
        fs_name = featureset.metadata.name
        merged_df = pd.merge(
            entity_df,
            featureset_df,
            on=indexes,
            how=self._join_type,
            left_on=left_keys,
            right_on=right_keys,
            suffixes=("", f"_{fs_name}"),
        )
        return merged_df
