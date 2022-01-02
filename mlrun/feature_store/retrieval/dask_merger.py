import dask.dataframe as dd
import pandas as pd
from dask.dataframe.multi import merge, merge_asof
from dask.distributed import Client

import mlrun

from ..feature_vector import OfflineVectorResponse
from .base import BaseMerger


class DaskFeatureMerger(BaseMerger):
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
    ):
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
        for name, columns in feature_set_fields.items():
            feature_set = feature_set_objects[name]
            feature_sets.append(feature_set)
            column_names = [name for name, alias in columns]
            df = feature_set.to_dataframe(
                columns=column_names,
                df_module=dd,
                start_time=start_time,
                end_time=end_time,
                time_column=entity_timestamp_column,
                index=False,
            )
            # rename columns with aliases
            df = df.rename(columns={name: alias for name, alias in columns if alias})

            df = df.persist()
            dfs.append(df)

        self.merge(entity_rows, entity_timestamp_column, feature_sets, dfs)

        self._result_df = self._result_df.drop(
            columns=self._drop_columns, errors="ignore"
        )

        if self.vector.status.label_column:
            self._result_df = self._result_df.dropna(
                subset=[self.vector.status.label_column]
            )

        if self._drop_indexes:
            self._result_df = self._result_df.reset_index(drop=True)
        self._write_to_target()

        # check if need to set indices
        self._result_df = self._set_indexes(self._result_df)
        return OfflineVectorResponse(self)

    def _reset_index(self, df):
        to_drop = df.index.name is None
        return df.reset_index(drop=to_drop)

    def _asof_join(
        self,
        entity_df,
        entity_timestamp_column: str,
        featureset,
        featureset_df: pd.DataFrame,
    ):
        indexes = list(featureset.spec.entities.keys())

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
            entity_df, featureset_df, left_index=True, right_index=True, by=indexes,
        )

        return merged_df

    def _join(
        self,
        entity_df,
        entity_timestamp_column: str,
        featureset,
        featureset_df: pd.DataFrame,
    ):
        indexes = list(featureset.spec.entities.keys())
        merged_df = merge(entity_df, featureset_df, on=indexes)
        return merged_df

    def get_status(self):
        if self._result_df is None:
            raise RuntimeError("unexpected status, no result df")
        return "completed"

    def get_df(self, to_pandas=True):
        if to_pandas and hasattr(self._result_df, "dask"):
            return self._result_df.compute()
        return self._result_df
