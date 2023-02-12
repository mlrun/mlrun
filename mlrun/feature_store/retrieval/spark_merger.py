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
import mlrun
from mlrun.datastore.targets import get_offline_target

from ...runtimes import RemoteSparkRuntime
from ...runtimes.sparkjob.abstract import AbstractSparkRuntime
from ..feature_vector import OfflineVectorResponse
from .base import BaseMerger


class SparkFeatureMerger(BaseMerger):
    engine = "spark"

    def __init__(self, vector, **engine_args):
        super().__init__(vector, **engine_args)
        self.spark = engine_args.get("spark", None)
        self.named_view = engine_args.get("named_view", False)
        self._pandas_df = None

    def to_spark_df(self, session, path):
        return session.read.load(path)

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
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col

        if self.spark is None:
            # create spark context
            self.spark = SparkSession.builder.appName(
                f"vector-merger-{self.vector.metadata.name}"
            ).getOrCreate()

        feature_sets = []
        dfs = []

        for name, columns in feature_set_fields.items():
            feature_set = feature_set_objects[name]
            feature_sets.append(feature_set)
            column_names = [name for name, alias in columns]

            if feature_set.spec.passthrough:
                if not feature_set.spec.source:
                    raise mlrun.errors.MLRunNotFoundError(
                        f"passthrough feature set {name} with no source"
                    )
                source_kind = feature_set.spec.source.kind
                source_path = feature_set.spec.source.path
            else:
                target = get_offline_target(feature_set)
                if not target:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"feature set {name} does not have offline targets"
                    )
                source_kind = target.kind
                source_path = target.get_target_path()

            # handling case where there are multiple feature sets and user creates vector where
            # entity_timestamp_column is from a specific feature set (can't be entity timestamp)
            source_driver = mlrun.datastore.sources.source_kind_to_driver[source_kind]
            if (
                entity_timestamp_column in column_names
                or feature_set.spec.timestamp_key == entity_timestamp_column
            ):
                source = source_driver(
                    name=self.vector.metadata.name,
                    path=source_path,
                    time_field=entity_timestamp_column,
                    start_time=start_time,
                    end_time=end_time,
                )
            else:
                source = source_driver(
                    name=self.vector.metadata.name,
                    path=source_path,
                    time_field=entity_timestamp_column,
                )

            # add the index/key to selected columns
            timestamp_key = feature_set.spec.timestamp_key

            df = source.to_spark_df(
                self.spark, named_view=self.named_view, time_field=timestamp_key
            )

            if timestamp_key and timestamp_key not in column_names:
                columns.append((timestamp_key, None))
            for entity in feature_set.spec.entities.keys():
                if entity not in column_names:
                    columns.append((entity, None))

            # select requested columns and rename with alias where needed
            df = df.select([col(name).alias(alias or name) for name, alias in columns])
            dfs.append(df)
            del df

        # convert pandas entity_rows to spark DF if needed
        if entity_rows is not None and not hasattr(entity_rows, "rdd"):
            entity_rows = self.spark.createDataFrame(entity_rows)

        # join the feature data frames
        self.merge(entity_rows, entity_timestamp_column, feature_sets, dfs)

        # filter joined data frame by the query param
        if query:
            self._result_df = self._result_df.filter(query)

        self._result_df = self._result_df.drop(*self._drop_columns)

        if self.vector.status.label_column:
            self._result_df = self._result_df.dropna(
                subset=[self.vector.status.label_column]
            )

        self._write_to_target()
        return OfflineVectorResponse(self)

    def _unpersist_df(self, df):
        df.unpersist()

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

        """Perform an as of join between entity and featureset.
        Join conditions:
        Args:
            entity_df (DataFrame): Spark dataframe representing the entities, to be joined with
                the feature tables.
            entity_timestamp_column (str): Column name in entity_df which represents
                event timestamp.
            featureset_df (Dataframe): Spark dataframe representing the feature table.
            featureset (FeatureSet): Feature set specification, which provides information on
                how the join should be performed, such as the entity primary keys.
        Returns:
            DataFrame: Join result, which contains all the original columns from entity_df, as well
                as all the features specified in featureset, where the feature columns will
                be prefixed with featureset_df name.
        """

        from pyspark.sql import Window
        from pyspark.sql.functions import col, monotonically_increasing_id, row_number

        entity_with_id = entity_df.withColumn("_row_nr", monotonically_increasing_id())
        indexes = list(featureset.spec.entities.keys())

        # get columns for projection
        projection = [
            col(col_name).alias(
                f"ft__{col_name}"
                if col_name in indexes + [entity_timestamp_column]
                else col_name
            )
            for col_name in featureset_df.columns
        ]

        aliased_featureset_df = featureset_df.select(projection)

        # set join conditions
        join_cond = (
            entity_with_id[entity_timestamp_column]
            >= aliased_featureset_df[f"ft__{entity_timestamp_column}"]
        )

        # join based on entities
        for key in indexes:
            join_cond = join_cond & (
                entity_with_id[key] == aliased_featureset_df[f"ft__{key}"]
            )

        conditional_join = entity_with_id.join(
            aliased_featureset_df, join_cond, "leftOuter"
        )
        for key in indexes + [entity_timestamp_column]:
            conditional_join = conditional_join.drop(
                aliased_featureset_df[f"ft__{key}"]
            )

        window = Window.partitionBy("_row_nr", *indexes).orderBy(
            col(entity_timestamp_column).desc(),
        )
        filter_most_recent_feature_timestamp = conditional_join.withColumn(
            "_rank", row_number().over(window)
        ).filter(col("_rank") == 1)

        return filter_most_recent_feature_timestamp.drop("_row_nr", "_rank")

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

        """
        spark dataframes join

        Args:
        entity_df (DataFrame): Spark dataframe representing the entities, to be joined with
            the feature tables.
        entity_timestamp_column (str): Column name in entity_df which represents
            event timestamp.
        featureset_df (Dataframe): Spark dataframe representing the feature table.
        featureset (FeatureSet): Feature set specification, which provide information on
            how the join should be performed, such as the entity primary keys.

        Returns:
            DataFrame: Join result, which contains all the original columns from entity_df, as well
                as all the features specified in featureset, where the feature columns will
                be prefixed with featureset_df name.

        """
        indexes = list(featureset.spec.entities.keys())
        merged_df = entity_df.join(featureset_df, on=indexes)
        return merged_df

    def get_df(self, to_pandas=True):
        if to_pandas:
            if self._pandas_df is None:
                self._pandas_df = self._result_df.toPandas()
                self._set_indexes(self._pandas_df)
            return self._pandas_df

        return self._result_df

    @classmethod
    def get_default_image(cls, kind):
        if kind == AbstractSparkRuntime.kind:
            return AbstractSparkRuntime._get_default_deployed_mlrun_image_name(
                with_gpu=False
            )
        elif kind == RemoteSparkRuntime.kind:
            return RemoteSparkRuntime.default_image
        else:
            raise mlrun.errors.MLRunInvalidArgumentError(f"Unsupported kind '{kind}'")
