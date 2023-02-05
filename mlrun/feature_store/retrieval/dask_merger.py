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
#
import re

import dask.dataframe as dd
from dask.dataframe.multi import merge, merge_asof
from dask.distributed import Client

import mlrun

from ..feature_vector import OfflineVectorResponse
from .base import BaseMerger


class DaskFeatureMerger(BaseMerger):
    engine = "dask"

    def __init__(self, vector, **engine_args):
        super().__init__(vector, **engine_args)
        self.client = engine_args.get("dask_client")
        self._dask_cluster_uri = engine_args.get("dask_cluster_uri")

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

    def _reset_index(self, df):
        to_drop = df.index.name is None
        df = df.reset_index(drop=to_drop)
        return df

    def _asof_join(
        self,
        entity_df,
        entity_timestamp_column: str,
        featureset,
        featureset_df,
        left_keys: list,
        right_keys: list,
    ):

        entity_df = self._reset_index(entity_df)
        entity_df = (
            entity_df
            if entity_timestamp_column not in entity_df
            else entity_df.set_index(entity_timestamp_column, drop=True)
        )
        featureset_df = self._reset_index(featureset_df)
        featureset_df = (
            featureset_df
            if entity_timestamp_column not in featureset_df
            else featureset_df.set_index(entity_timestamp_column, drop=True)
        )

        merged_df = merge_asof(
            entity_df,
            featureset_df,
            left_index=True,
            right_index=True,
            left_by=left_keys or None,
            right_by=right_keys or None,
            suffixes=("", f"_{featureset.metadata.name}_"),
        )
        for col in merged_df.columns:
            if re.findall(f"_{featureset.metadata.name}_$", col):
                self._append_drop_column(col)

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
        merged_df = merge(
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

    def get_status(self):
        if self._result_df is None:
            raise RuntimeError("unexpected status, no result df")
        return "completed"

    def get_df(self, to_pandas=True):
        if to_pandas and hasattr(self._result_df, "dask"):
            df = self._result_df.compute()
        else:
            df = self._result_df
        self._set_indexes(df)
        return df

    def create_engine_env(self):
        if "index" not in self._index_columns:
            self._append_drop_column("index")

        # init the dask client if needed
        if not self.client:
            if self._dask_cluster_uri:
                function = mlrun.import_function(self._dask_cluster_uri)
                self.client = function.client
            else:
                self.client = Client()

    def get_engine_df(
        self,
        feature_set,
        feature_set_name,
        column_names,
        start_time,
        end_time,
        entity_timestamp_column,
    ):
        df = feature_set.to_dataframe(
            columns=column_names,
            df_module=dd,
            start_time=start_time,
            end_time=end_time,
            time_column=entity_timestamp_column,
            index=False,
        )

        return self._reset_index(df).persist()

    def rename_columns_and_select(self, df, rename_col_dict, all_columns):
        return df.rename(
            columns=rename_col_dict,
        )

    def drop_columns_from_result(self):
        self._result_df = self._result_df.drop(
            columns=self._drop_columns, errors="ignore"
        )

    def filter(self, query):
        self._result_df = self._result_df.query(query)

    def orderBy(self, order_by_active):
        self._result_df.sort_values(by=order_by_active)
