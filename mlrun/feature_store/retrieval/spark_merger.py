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


import mlrun
from mlrun.data_types.to_pandas import spark_df_to_pandas
from mlrun.datastore.sources import ParquetSource
from mlrun.datastore.targets import get_offline_target
from mlrun.runtimes import RemoteSparkRuntime
from mlrun.runtimes.sparkjob import Spark3Runtime
from mlrun.utils.helpers import additional_filters_warning

from .base import BaseMerger


class SparkFeatureMerger(BaseMerger):
    engine = "spark"
    support_offline = True

    def __init__(self, vector, **engine_args):
        super().__init__(vector, **engine_args)
        self.spark = engine_args.get("spark", None)
        self.named_view = engine_args.get("named_view", False)
        self._pandas_df = None

    def to_spark_df(self, session, path):
        return session.read.load(path)

    def _unpersist_df(self, df):
        df.unpersist()

    def _asof_join(
        self,
        entity_df,
        entity_timestamp_column: str,
        featureset_name: str,
        featureset_timstamp: str,
        featureset_df: list,
        left_keys: list,
        right_keys: list,
    ):
        """Perform an as of join between entity and featureset.
        Join conditions:
        Args:
            entity_df (DataFrame): Spark dataframe representing the entities, to be joined with
                the feature tables.
            entity_timestamp_column (str): Column name in entity_df which represents
                event timestamp.
            featureset (Dataframe): Spark dataframe representing the feature table.
            featureset (FeatureSet): Feature set specification, which provides information on
                how the join should be performed, such as the entity primary keys.
        Returns:
            DataFrame: Join result, which contains all the original columns from entity_df, as well
                as all the features specified in featureset, where the feature columns will
                be prefixed with featureset_df name.
                :param featureset_name:
                :param featureset_timstamp:
        """

        from pyspark.sql import Window
        from pyspark.sql.functions import col, monotonically_increasing_id, row_number

        entity_with_id = entity_df.withColumn("_row_nr", monotonically_increasing_id())
        rename_right_keys = {}
        for key in right_keys + [featureset_timstamp]:
            if key in entity_df.columns:
                rename_right_keys[key] = f"ft__{key}"
        # get columns for projection
        projection = [
            col(col_name).alias(rename_right_keys.get(col_name, col_name))
            for col_name in featureset_df.columns
        ]

        aliased_featureset_df = featureset_df.select(projection)
        right_timestamp = rename_right_keys.get(
            featureset_timstamp, featureset_timstamp
        )

        # set join conditions
        join_cond = (
            entity_with_id[entity_timestamp_column]
            >= aliased_featureset_df[right_timestamp]
        )

        # join based on entities
        for key_l, key_r in zip(left_keys, right_keys):
            join_cond = join_cond & (
                entity_with_id[key_l]
                == aliased_featureset_df[rename_right_keys.get(key_r, key_r)]
            )

        conditional_join = entity_with_id.join(
            aliased_featureset_df, join_cond, "leftOuter"
        )

        window = Window.partitionBy("_row_nr").orderBy(
            col(right_timestamp).desc(),
        )
        filter_most_recent_feature_timestamp = conditional_join.withColumn(
            "_rank", row_number().over(window)
        ).filter(col("_rank") == 1)

        for key in right_keys + [featureset_timstamp]:
            if key in entity_df.columns + [entity_timestamp_column]:
                filter_most_recent_feature_timestamp = (
                    filter_most_recent_feature_timestamp.drop(
                        aliased_featureset_df[f"ft__{key}"]
                    )
                )
        return filter_most_recent_feature_timestamp.drop("_row_nr", "_rank").orderBy(
            col(entity_timestamp_column)
        )

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
        """
        spark dataframes join

        :param entity_df (DataFrame): Spark dataframe representing the entities, to be joined with
            the feature tables.
        :param entity_timestamp_column (str): Column name in entity_df which represents
            event timestamp.
        :param featureset_df (Dataframe): Spark dataframe representing the feature table.
        :param featureset_name:
        :param featureset_timestamp:

        Returns:
            DataFrame: Join result, which contains all the original columns from entity_df, as well
                as all the features specified in featureset, where the feature columns will
                be prefixed with featureset_df name.
        """
        if left_keys != right_keys:
            join_cond = [
                entity_df[key_l] == featureset_df[key_r]
                for key_l, key_r in zip(left_keys, right_keys)
            ]
        else:
            join_cond = left_keys

        merged_df = entity_df.join(
            featureset_df,
            join_cond,
            how=self._join_type,
        )
        return merged_df

    def get_df(self, to_pandas=True):
        if to_pandas:
            if self._pandas_df is None:
                df = spark_df_to_pandas(self._result_df)
                self._pandas_df = df
                self._set_indexes(self._pandas_df)
            return self._pandas_df

        return self._result_df

    @classmethod
    def get_default_image(cls, kind):
        if kind == Spark3Runtime.kind:
            return Spark3Runtime._get_default_deployed_mlrun_image_name(with_gpu=False)
        elif kind == RemoteSparkRuntime.kind:
            return RemoteSparkRuntime.default_image
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(f"Unsupported kind '{kind}'")

    def _create_engine_env(self):
        from pyspark.sql import SparkSession

        if self.spark is None:
            # create spark context
            self.spark = (
                SparkSession.builder.appName(
                    f"vector-merger-{self.vector.metadata.name}"
                )
                .config("spark.driver.memory", "2g")
                .getOrCreate()
            )

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
        mlrun.utils.helpers.additional_filters_warning(
            additional_filters, self.__class__
        )

        source_kwargs = {}
        if feature_set.spec.passthrough:
            if not feature_set.spec.source:
                raise mlrun.errors.MLRunNotFoundError(
                    f"passthrough feature set {feature_set_name} with no source"
                )
            source_kind = feature_set.spec.source.kind
            source_path = feature_set.spec.source.path
            source_kwargs.update(feature_set.spec.source.attributes)
            source_kwargs.pop("additional_filters", None)
        else:
            target = get_offline_target(feature_set)
            if not target:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"feature set {feature_set_name} does not have offline targets"
                )
            source_kind = target.kind
            source_path = target.get_target_path()
            source_kwargs = target.source_spark_attributes
        # handling case where there are multiple feature sets and user creates vector where
        # entity_timestamp_column is from a specific feature set (can't be entity timestamp)
        source_driver = mlrun.datastore.sources.source_kind_to_driver[source_kind]

        if source_driver != ParquetSource:
            additional_filters_warning(additional_filters, source_driver)
            additional_filters = None
        additional_filters_dict = (
            {"additional_filters": additional_filters} if additional_filters else {}
        )
        source = source_driver(
            name=self.vector.metadata.name,
            path=source_path,
            time_field=time_column,
            start_time=start_time,
            end_time=end_time,
            **additional_filters_dict,
            **source_kwargs,
        )

        columns = column_names + [ent.name for ent in feature_set.spec.entities]
        if (
            feature_set.spec.timestamp_key
            and feature_set.spec.timestamp_key not in columns
        ):
            columns.append(feature_set.spec.timestamp_key)

        return source.to_spark_df(
            self.spark,
            named_view=self.named_view,
            time_field=time_column,
            columns=columns,
        )

    def _rename_columns_and_select(
        self,
        df,
        rename_col_dict,
        columns=None,
    ):
        from pyspark.sql.functions import col

        return df.select(
            [
                col(name).alias(rename_col_dict.get(name, name))
                for name in columns or rename_col_dict.keys()
            ]
        )

    def _drop_columns_from_result(self):
        self._result_df = self._result_df.drop(*self._drop_columns)

    def _filter(self, query):
        self._result_df = self._result_df.filter(query)

    def _order_by(self, order_by_active):
        from pyspark.sql.functions import col

        self._result_df = self._result_df.orderBy(
            *[col(col_name).asc_nulls_last() for col_name in order_by_active]
        )

    def _convert_entity_rows_to_engine_df(self, entity_rows):
        if entity_rows is not None and not hasattr(entity_rows, "rdd"):
            return self.spark.createDataFrame(entity_rows)
        return entity_rows
