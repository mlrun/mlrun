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
        order_by=None,
    ):
        return super()._generate_vector(
            entity_rows,
            entity_timestamp_column,
            feature_set_objects,
            feature_set_fields,
            start_time=start_time,
            end_time=end_time,
            query=query,
            order_by=order_by,
        )

    def _asof_join(
        self,
        entity_df,
        entity_timestamp_column: str,
        featureset,
        featureset_df,
        left_keys: list,
        right_keys: list,
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

    def create_engine_env(self):
        pass

    def get_engine_df(
        self,
        feature_set,
        feature_set_name,
        column_names,
        start_time,
        end_time,
        entity_timestamp_column,
    ):
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
        return df

    def rename_columns_and_select(self, df, rename_col_dict, all_columns):
        df.rename(
            columns=rename_col_dict,
            inplace=True,
        )

    def drop_columns_from_result(self):
        self._result_df.drop(columns=self._drop_columns, inplace=True, errors="ignore")

    def filter(self, query):
        self._result_df.query(query, inplace=True)

    def orderBy(self, order_by_active):
        self._result_df.sort_values(by=order_by_active, ignore_index=True, inplace=True)
