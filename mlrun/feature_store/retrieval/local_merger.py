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
    ):

        feature_sets = []
        dfs = []
        for name, columns in feature_set_fields.items():
            feature_set = feature_set_objects[name]
            feature_sets.append(feature_set)
            column_names = [name for name, alias in columns]
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
                    columns=column_names, time_column=entity_timestamp_column,
                )
            # rename columns with aliases
            df.rename(
                columns={name: alias for name, alias in columns if alias}, inplace=True
            )
            dfs.append(df)

        self.merge(entity_rows, entity_timestamp_column, feature_sets, dfs)

        self._result_df.drop(columns=self._drop_columns, inplace=True, errors="ignore")

        if self.vector.status.label_column:
            self._result_df = self._result_df.dropna(
                subset=[self.vector.status.label_column]
            )

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
        featureset_df: pd.DataFrame,
    ):
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
        featureset_df: pd.DataFrame,
    ):
        indexes = list(featureset.spec.entities.keys())
        merged_df = pd.merge(entity_df, featureset_df, on=indexes)
        return merged_df
