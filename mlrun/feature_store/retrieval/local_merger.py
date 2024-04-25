# Copyright 2023 Iguazio
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

from .base import BaseMerger


class LocalFeatureMerger(BaseMerger):
    engine = "local"
    support_offline = True

    def __init__(self, vector, **engine_args):
        super().__init__(vector, **engine_args)

    def _asof_join(
        self,
        entity_df,
        entity_timestamp_column: str,
        featureset_name,
        featureset_timstamp,
        featureset_df,
        left_keys: list,
        right_keys: list,
    ):
        index_col_not_in_entity = "index" not in entity_df.columns
        index_col_not_in_featureset = "index" not in featureset_df.columns
        entity_df[entity_timestamp_column] = pd.to_datetime(
            entity_df[entity_timestamp_column]
        )
        featureset_df[featureset_timstamp] = pd.to_datetime(
            featureset_df[featureset_timstamp]
        )
        entity_df.sort_values(by=entity_timestamp_column, inplace=True)
        featureset_df.sort_values(by=featureset_timstamp, inplace=True)

        featureset_df = self._normalize_timestamp_column(
            entity_timestamp_column,
            entity_df,
            featureset_timstamp,
            featureset_df,
            featureset_name,
        )

        merged_df = pd.merge_asof(
            entity_df,
            featureset_df,
            left_on=entity_timestamp_column,
            right_on=featureset_timstamp,
            left_by=left_keys or None,
            right_by=right_keys or None,
            suffixes=("", f"_{featureset_name}_"),
        )
        for col in merged_df.columns:
            if re.findall(f"_{featureset_name}_$", col):
                self._append_drop_column(col)
        # Undo indexing tricks for asof merge
        # to return the correct indexes and not
        # overload `index` columns
        if (
            "index" not in left_keys
            and "index" not in right_keys
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
        featureset_name,
        featureset_timestamp,
        featureset_df,
        left_keys: list,
        right_keys: list,
    ):
        merged_df = pd.merge(
            entity_df,
            featureset_df,
            how=self._join_type,
            left_on=left_keys,
            right_on=right_keys,
            suffixes=("", f"_{featureset_name}_"),
        )
        for col in merged_df.columns:
            if re.findall(f"_{featureset_name}_$", col):
                self._append_drop_column(col)
        return merged_df

    def _create_engine_env(self):
        pass

    def _get_engine_df(
        self,
        feature_set,
        feature_set_name,
        column_names=None,
        start_time=None,
        end_time=None,
        time_column=None,
        additional_filters=None,
    ):
        df = feature_set.to_dataframe(
            columns=column_names,
            start_time=start_time,
            end_time=end_time,
            time_column=time_column,
            additional_filters=additional_filters,
        )
        if df.index.names[0]:
            df.reset_index(inplace=True)
        return df

    def _rename_columns_and_select(self, df, rename_col_dict, columns=None):
        df.rename(
            columns=rename_col_dict,
            inplace=True,
        )

    def _drop_columns_from_result(self):
        self._result_df.drop(columns=self._drop_columns, inplace=True, errors="ignore")

    def _filter(self, query):
        self._result_df.query(query, inplace=True)

    def _order_by(self, order_by_active):
        self._result_df.sort_values(by=order_by_active, ignore_index=True, inplace=True)

    def _convert_entity_rows_to_engine_df(self, entity_rows):
        return entity_rows
