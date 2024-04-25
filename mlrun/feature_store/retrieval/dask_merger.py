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
#
import re

import mlrun

from .base import BaseMerger


class DaskFeatureMerger(BaseMerger):
    engine = "dask"
    support_offline = True

    def __init__(self, vector, **engine_args):
        super().__init__(vector, **engine_args)
        try:
            import dask  # noqa: F401
        except (ModuleNotFoundError, ImportError) as exc:
            raise ImportError(
                "Using 'DaskFeatureMerger' requires dask package. Use pip install mlrun[dask] to install it."
            ) from exc

        self.client = engine_args.get("dask_client")
        self._dask_cluster_uri = engine_args.get("dask_cluster_uri")

    def _reset_index(self, df):
        to_drop = df.index.name is None
        df = df.reset_index(drop=to_drop)
        return df

    def _asof_join(
        self,
        entity_df,
        entity_timestamp_column: str,
        featureset_name: str,
        featureset_timestamp: str,
        featureset_df: list,
        left_keys: list,
        right_keys: list,
    ):
        from dask.dataframe.multi import merge_asof

        featureset_df = self._normalize_timestamp_column(
            entity_timestamp_column,
            entity_df,
            featureset_timestamp,
            featureset_df,
            featureset_name,
        )

        def sort_partition(partition, timestamp):
            return partition.sort_values(timestamp)

        entity_df = entity_df.map_partitions(
            sort_partition, timestamp=entity_timestamp_column
        )
        featureset_df = featureset_df.map_partitions(
            sort_partition, timestamp=featureset_timestamp
        )

        merged_df = merge_asof(
            entity_df,
            featureset_df,
            left_on=entity_timestamp_column,
            right_on=featureset_timestamp,
            left_by=left_keys or None,
            right_by=right_keys or None,
            suffixes=("", f"_{featureset_name}_"),
        )
        for col in merged_df.columns:
            if re.findall(f"_{featureset_name}_$", col):
                self._append_drop_column(col)

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
        from dask.dataframe.multi import merge

        merged_df = merge(
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

    def _create_engine_env(self):
        from dask.distributed import Client

        if "index" not in self._index_columns:
            self._append_drop_column("index")

        # init the dask client if needed
        if not self.client:
            if self._dask_cluster_uri:
                function = mlrun.import_function(self._dask_cluster_uri)
                self.client = function.client
            else:
                self.client = Client()

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
        import dask.dataframe as dd

        df = feature_set.to_dataframe(
            columns=column_names,
            df_module=dd,
            start_time=start_time,
            end_time=end_time,
            time_column=time_column,
            index=False,
            additional_filters=additional_filters,
        )

        return self._reset_index(df).persist()

    def _rename_columns_and_select(self, df, rename_col_dict, columns=None):
        return df.rename(
            columns=rename_col_dict,
        )

    def _drop_columns_from_result(self):
        self._result_df = self._result_df.drop(
            columns=self._drop_columns, errors="ignore"
        )

    def _filter(self, query):
        self._result_df = self._result_df.query(query)

    def _order_by(self, order_by_active):
        self._result_df.sort_values(by=order_by_active)

    def _convert_entity_rows_to_engine_df(self, entity_rows):
        import dask.dataframe as dd

        if entity_rows is not None and not hasattr(entity_rows, "dask"):
            return dd.from_pandas(entity_rows, npartitions=len(entity_rows.columns))

        return entity_rows
