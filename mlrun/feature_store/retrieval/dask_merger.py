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
    ):
        if "index" not in self._index_columns:
            self._append_drop_column("index")

        # init the dask client if needed
        if not self.client:
            if self._dask_cluster_uri:
                function = mlrun.import_function(self._dask_cluster_uri)
                self.client = function.client
            else:
                self.client = Client()

        # load dataframes
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

            df = feature_set.to_dataframe(
                columns=column_names,
                df_module=dd,
                start_time=start_time,
                end_time=end_time,
                time_column=entity_timestamp_column,
                index=False,
            )

            df = df.reset_index()
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

            df = df.persist()

            # rename columns to be unique for each feature set
            rename_col_dict = {
                col: f"{col}_{name}"
                for col in column_names
                if col not in node.data["save_cols"]
            }
            df = df.rename(
                columns=rename_col_dict,
            )

            dfs.append(df)
            del df

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

        self._result_df = self._result_df.drop(
            columns=self._drop_columns, errors="ignore"
        )

        # renaming all columns according to self._alias
        self._result_df = self._result_df.rename(
            columns=self._alias,
        )

        if self.vector.status.label_column:
            self._result_df = self._result_df.dropna(
                subset=[self.vector.status.label_column]
            )
        # filter joined data frame by the query param
        if query:
            self._result_df = self._result_df.query(query)

        if self._drop_indexes:
            self._result_df = self._reset_index(self._result_df)
        else:
            self._result_df = self._set_indexes(self._result_df)
        self._write_to_target()

        return OfflineVectorResponse(self)

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
        columns: list,
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
        columns: list,
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
            return self._result_df.compute()
        return self._result_df
