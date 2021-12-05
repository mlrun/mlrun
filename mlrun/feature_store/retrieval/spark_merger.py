import mlrun
from mlrun.datastore.targets import get_offline_target

from ..feature_vector import OfflineVectorResponse
from .base import BaseMerger


class SparkFeatureMerger(BaseMerger):
    def __init__(self, vector, **engine_args):
        super().__init__(vector, **engine_args)
        self.spark = engine_args.get("spark", None)
        self.named_view = engine_args.get("named_view", False)

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
            target = get_offline_target(feature_set)
            if not target:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Feature set {name} does not have offline targets"
                )

            # handling case where there are multiple feature sets and user creates vector where
            # entity_timestamp_column is from a specific feature set (can't be entity timestamp)
            source_driver = mlrun.datastore.sources.source_kind_to_driver[target.kind]
            if (
                entity_timestamp_column in column_names
                or feature_set.spec.timestamp_key == entity_timestamp_column
            ):
                source = source_driver(
                    self.vector.metadata.name,
                    target.path,
                    time_field=entity_timestamp_column,
                    start_time=start_time,
                    end_time=end_time,
                )
            else:
                source = source_driver(
                    self.vector.metadata.name,
                    target.path,
                    time_field=entity_timestamp_column,
                )

            df = source.to_spark_df(self.spark, named_view=self.named_view)
            timestamp_key = feature_set.spec.timestamp_key
            if timestamp_key and timestamp_key not in column_names:
                columns.append((timestamp_key, None))
            for entity in feature_set.spec.entities.keys():
                if entity not in column_names:
                    columns.append((entity, None))

            print("extended cols:", columns)
            df = df.select([col(name).alias(alias or name) for name, alias in columns])
            df.show()
            # df = df.select(column_names)

            dfs.append(df)

        if entity_rows is not None and not hasattr(entity_rows, "rdd"):
            # convert pandas to spark DF if needed
            entity_rows = self.spark.createDataFrame(entity_rows)

        self.merge(entity_rows, entity_timestamp_column, feature_sets, dfs)

        self._result_df = self._result_df.drop(*self._drop_columns)

        # todo: drop rows with null label_column values

        self._write_to_target()

        # todo: drop/set indexes if needed

        return OfflineVectorResponse(self)

    def merge(
        self,
        entity_df,
        entity_timestamp_column: str,
        featuresets: list,
        featureset_dfs: list,
    ):
        merged_df = entity_df
        if entity_df is None and featureset_dfs:
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
        self, entity_df, entity_timestamp_column: str, featureset, featureset_df,
    ):

        """Perform an as of join between entity and featureset.
        Join conditions:
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

        from pyspark.sql import Window
        from pyspark.sql.functions import col, monotonically_increasing_id, row_number

        entity_with_id = entity_df.withColumn("_row_nr", monotonically_increasing_id())

        # set timestamp column
        if entity_timestamp_column is None:
            entity_timestamp_column = (
                entity_timestamp_column or featureset.spec.timestamp_key
            )

        # name featureset col with prefix
        feature_event_timestamp_column_with_prefix = (
            f"{'ft'}__{entity_timestamp_column}"
        )

        # get columns for projection
        projection = [
            col(col_name).alias(f"{'ft'}__{col_name}")
            for col_name in featureset_df.columns
        ]

        aliased_feature_table_df = featureset_df.select(projection)
        print("aliased:")
        aliased_feature_table_df.show()

        # set join conditions
        print(entity_with_id.columns)
        join_cond = (
            entity_with_id[entity_timestamp_column]
            >= aliased_feature_table_df[feature_event_timestamp_column_with_prefix]
        )

        # join based on entities
        for key in [d["name"] for d in featureset.spec.to_dict()["entities"]]:
            join_cond = join_cond & (
                entity_with_id[key] == aliased_feature_table_df[f"{'ft'}__{key}"]
            )

        conditional_join = entity_with_id.join(
            aliased_feature_table_df, join_cond, "leftOuter"
        )
        for key in [d["name"] for d in featureset.spec.to_dict()["entities"]]:
            conditional_join = conditional_join.drop(
                aliased_feature_table_df[f"{'ft'}__{key}"]
            )

        window = Window.partitionBy(
            "_row_nr", *[d["name"] for d in featureset.spec.to_dict()["entities"]]
        ).orderBy(col(feature_event_timestamp_column_with_prefix).desc(),)
        filter_most_recent_feature_timestamp = conditional_join.withColumn(
            "_rank", row_number().over(window)
        ).filter(col("_rank") == 1)

        return filter_most_recent_feature_timestamp.select(
            entity_df.columns
            + ["ft__" + d["name"] for d in featureset.spec.to_dict()["features"]]
            + ["ft__" + entity_timestamp_column]
        ).show(truncate=False)

    def _join(
        self, entity_df, entity_timestamp_column: str, featureset, featureset_df,
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
