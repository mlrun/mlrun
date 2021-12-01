from mlrun.datastore.targets import get_offline_target
from mlrun.datastore.utils import store_path_to_spark

from ..feature_vector import OfflineVectorResponse
from .base import BaseMerger


class SparkFeatureMerger(BaseMerger):
    def __init__(self, vector, **engine_args):
        self._result_df = None
        self.vector = vector
        self.spark = engine_args.get("spark", None)

    def to_spark_df(self, session, path):
        return session.read.load(path)

    def start(
        self,
        entity_rows=None,
        entity_timestamp_column=None,
        target=None,
        drop_columns=None,
        start_time=None,
        end_time=None,
        with_indexes=None,
        update_stats=None,
    ):
        from pyspark.sql import SparkSession

        if self.spark is None:
            # create spark context
            self.spark = SparkSession.builder.appName("name").getOrCreate()

        if not drop_columns:
            drop_columns = []
        index_columns = []
        drop_indexes = False if self.vector.spec.with_indexes else True

        def append_drop_column(key):
            if key and key not in drop_columns:
                drop_columns.append(key)

        def append_index(key):
            if key:
                if key not in index_columns:
                    index_columns.append(key)
                if drop_indexes:
                    append_drop_column(key)

        if entity_timestamp_column and drop_indexes:
            drop_columns.append(entity_timestamp_column)
        feature_set_objects, feature_set_fields = self.vector.parse_features()
        feature_sets = []
        dfs = []

        for name, columns in feature_set_fields.items():
            feature_set = feature_set_objects[name]
            feature_sets.append(feature_set)
            column_names = [name for name, alias in columns]
            entities = list(feature_set.spec.entities.keys())
            target = get_offline_target(feature_set)
            if not target:
                # TODO - throw
                break
            path = target.path
            df = self.spark.read.load(store_path_to_spark(path))
            dfs.append(df)

        self.merge(entity_rows, entity_timestamp_column, feature_sets, dfs)
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
