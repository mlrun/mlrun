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

import mlrun
from mlrun.datastore.targets import get_offline_target

from ...runtimes import RemoteSparkRuntime
from ...runtimes.sparkjob.abstract import AbstractSparkRuntime
from ..feature_vector import OfflineVectorResponse
from .base import BaseMerger


class SparkFeatureMerger(BaseMerger):
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

            for column in node.data["save_cols"]:
                if column not in column_names:
                    self._append_drop_column(column)
            column_names += node.data["save_cols"]

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

            if not entity_timestamp_column:
                entity_timestamp_column = feature_set.spec.timestamp_key
            # add the index/key to selected columns
            timestamp_key = feature_set.spec.timestamp_key

            df = source.to_spark_df(
                self.spark, named_view=self.named_view, time_field=timestamp_key
            )

            print("before select")
            print(df.show())
            column_names += node.data["save_index"]
            node.data["save_cols"] += node.data["save_index"]
            if feature_set.spec.timestamp_key:
                entity_timestamp_column_list = [feature_set.spec.timestamp_key]
                column_names += entity_timestamp_column_list
                node.data["save_cols"] += entity_timestamp_column_list

            # rename columns to be unique for each feature set
            rename_col_dict = {
                column: f"{column}_{name}"
                for column in column_names
                if column not in node.data["save_cols"]
            }
            fs_entities = list(feature_set.spec.entities.keys())
            # select requested columns and rename with alias where needed

            df = df.select(
                [
                    col(name).alias(rename_col_dict.get(name, name))
                    for name in list(set(column_names + fs_entities))
                ]
            )

            print("after select")
            print(df.show())
            dfs.append(df)
            del df

            keys.append([node.data["left_keys"], node.data["right_keys"]])

            # update alias according to the unique column name
            new_columns = []
            if not self._drop_indexes:
                new_columns.extend([(ind, ind) for ind in fs_entities])
            for column, alias in columns:
                if column in rename_col_dict and alias:
                    new_columns.append((rename_col_dict[column], alias))
                elif column in rename_col_dict and not alias:
                    new_columns.append((rename_col_dict[column], column))
                else:
                    new_columns.append((column, alias))
            all_columns.append(new_columns)
            self._update_alias(
                dictionary={name: alias for name, alias in new_columns if alias}
            )

        # convert pandas entity_rows to spark DF if needed
        if entity_rows is not None and not hasattr(entity_rows, "rdd"):
            entity_rows = self.spark.createDataFrame(entity_rows)

        # join the feature data frames
        self.merge(
            entity_df=entity_rows,
            entity_timestamp_column=entity_timestamp_column,
            featuresets=feature_sets,
            featureset_dfs=dfs,
            keys=keys,
            all_columns=all_columns,
        )

        if not self._drop_indexes and entity_timestamp_column:
            self._update_alias(key=entity_timestamp_column, val=entity_timestamp_column)
        self._result_df = self._result_df.drop(*self._drop_columns)

        print("before last select")
        print(self._result_df.show())
        self._result_df = self._result_df.select(
            [col(name).alias(alias or name) for name, alias in self._alias.items()]
        )

        if self.vector.status.label_column:
            self._result_df = self._result_df.dropna(
                subset=[self.vector.status.label_column]
            )
        # filter joined data frame by the query param
        if query:
            self._result_df = self._result_df.filter(query)

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
        # indexes = list(featureset.spec.entities.keys())
        rename_right_keys = {}
        for key in right_keys + [entity_timestamp_column]:
            if key in entity_df.columns:
                rename_right_keys[key] = f"ft__{key}"
        # get columns for projection
        projection = [
            col(col_name).alias(
                rename_right_keys.get(col_name, col_name)
            )
            for col_name in featureset_df.columns
        ]

        aliased_featureset_df = featureset_df.select(projection)

        # set join conditions
        join_cond = (
            entity_with_id[entity_timestamp_column]
            >= aliased_featureset_df[rename_right_keys.get(entity_timestamp_column, entity_timestamp_column)]
        )

        # join based on entities
        for key_l, key_r in zip(left_keys, right_keys):
            join_cond = join_cond & (
                entity_with_id[key_l] == aliased_featureset_df[rename_right_keys.get(key_r, key_r)]
            )

        conditional_join = entity_with_id.join(
            aliased_featureset_df, join_cond, "leftOuter"
        )
        for key in right_keys + [entity_timestamp_column]:
            if f"ft__{key}" in conditional_join.columns:
                conditional_join = conditional_join.drop(
                    aliased_featureset_df[f"ft__{key}"]
                )

        window = Window.partitionBy("_row_nr", *left_keys).orderBy(
            col(entity_timestamp_column).desc(),
        )
        filter_most_recent_feature_timestamp = conditional_join.withColumn(
            "_rank", row_number().over(window)
        ).filter(col("_rank") == 1)

        return filter_most_recent_feature_timestamp.drop("_row_nr", "_rank").orderBy(col(entity_timestamp_column))

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
        # fs_name = featureset.metadata.name
        join_cond = None
        print(entity_df)
        print(featureset_df)

        for key_l, key_r in zip(left_keys, right_keys):
            join_cond = join_cond & (
                entity_df[key_l] == featureset_df[key_r]
            ) if join_cond else (
                entity_df[key_l] == featureset_df[key_r]
            )


        merged_df = entity_df.join(
            featureset_df,
            join_cond,
            how=self._join_type,
        )
        # repeated_columns = [c for c in entity_df.columns if c in featureset_df.columns]
        # for col in repeated_columns:
        #     merged_df = merged_df.drop(featureset_df[col])
        # # for col in merged_df.columns:
        # #     if re.findall(f"_{fs_name}_$", col):
        # #         self._append_drop_column(col)
        print(merged_df)
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
