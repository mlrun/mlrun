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

from typing import List

import pandas as pd

import mlrun
import mlrun.errors

from ...utils import logger
from ..feature_vector import OfflineVectorResponse


class LocalFeatureMerger:
    def __init__(self, vector):
        self._result_df = None
        self.vector = vector

    def start(
        self,
        entity_rows=None,
        entity_timestamp_column=None,
        target=None,
        drop_columns=None,
        start_time=None,
        end_time=None,
    ):
        index_columns = []
        drop_indexes = False if self.vector.spec.with_indexes else True

        def append_index(key):
            if drop_indexes and key and key not in index_columns:
                index_columns.append(key)

        if entity_timestamp_column:
            index_columns.append(entity_timestamp_column)
        feature_set_objects, feature_set_fields = self.vector.parse_features()
        self.vector.save()

        # load dataframes
        feature_sets = []
        dfs = []
        df_module = None  # for use of dask or other non pandas df module
        for name, columns in feature_set_fields.items():
            feature_set = feature_set_objects[name]
            feature_sets.append(feature_set)
            column_names = [name for name, alias in columns]
            df = feature_set.to_dataframe(
                columns=column_names,
                df_module=df_module,
                start_time=start_time,
                end_time=end_time,
                time_column=entity_timestamp_column,
            )
            # rename columns with aliases
            df.rename(
                columns={name: alias for name, alias in columns if alias}, inplace=True
            )
            dfs.append(df)
            append_index(feature_set.spec.timestamp_key)
            for key in feature_set.spec.entities.keys():
                append_index(key)

        self.merge(entity_rows, entity_timestamp_column, feature_sets, dfs)
        if drop_columns or index_columns:
            for field in drop_columns or []:
                if field not in index_columns:
                    index_columns.append(field)

            self._result_df.drop(columns=index_columns, inplace=True, errors="ignore")

        if target:
            is_persistent_vector = self.vector.metadata.name is not None
            if not target.path and not is_persistent_vector:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "target path was not specified"
                )
            target.set_resource(self.vector)
            size = target.write_dataframe(self._result_df)
            if is_persistent_vector:
                target_status = target.update_resource_status("ready", size=size)
                logger.info(f"wrote target: {target_status}")
                self.vector.save()
        return OfflineVectorResponse(self)

    def merge(
        self,
        entity_df,
        entity_timestamp_column: str,
        featuresets: list,
        featureset_dfs: List[pd.DataFrame],
    ):
        merged_df = entity_df
        if entity_df is None:
            merged_df = featureset_dfs.pop(0)
            featureset = featuresets.pop(0)
            entity_timestamp_column = (
                entity_timestamp_column or featureset.spec.timestamp_key
            )

        for featureset, featureset_df in zip(featuresets, featureset_dfs):
            if featureset.spec.timestamp_key:
                merge_func = self._asof_join
            else:
                merge_func = self._join

            merged_df = merge_func(
                merged_df, entity_timestamp_column, featureset, featureset_df,
            )

        self._result_df = merged_df

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

    def get_status(self):
        if self._result_df is None:
            raise RuntimeError("unexpected status, no result df")
        return "completed"

    def get_df(self):
        return self._result_df
