import abc

import mlrun
from mlrun.datastore.targets import CSVTarget, ParquetTarget

from ...utils import logger


class BaseMerger(abc.ABC):
    """abstract feature merger class"""

    def __init__(self, vector, **engine_args):
        self.vector = vector

        self._result_df = None
        self._drop_columns = []
        self._index_columns = []
        self._drop_indexes = True
        self._target = None

    def _append_drop_column(self, key):
        if key and key not in self._drop_columns:
            self._drop_columns.append(key)

    def _append_index(self, key):
        if key:
            if key not in self._index_columns:
                self._index_columns.append(key)
            if self._drop_indexes:
                self._append_drop_column(key)

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
        self._target = target

        # calculate the index columns and columns we need to drop
        self._drop_columns = drop_columns or self._drop_columns
        if self.vector.spec.with_indexes or with_indexes:
            self._drop_indexes = False

        if entity_timestamp_column and self._drop_indexes:
            self._append_drop_column(entity_timestamp_column)

        # retrieve the feature set objects/fields needed for the vector
        feature_set_objects, feature_set_fields = self.vector.parse_features(
            update_stats=update_stats
        )
        if len(feature_set_fields) == 0:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "No features in vector. Make sure to infer the schema on all the feature sets first"
            )

        if update_stats:
            # update the feature vector objects with refreshed stats
            self.vector.save()

        for feature_set in feature_set_objects.values():
            if not entity_timestamp_column and self._drop_indexes:
                self._append_drop_column(feature_set.spec.timestamp_key)
            for key in feature_set.spec.entities.keys():
                self._append_index(key)

        return self._generate_vector(
            entity_rows,
            entity_timestamp_column,
            feature_set_objects=feature_set_objects,
            feature_set_fields=feature_set_fields,
            start_time=start_time,
            end_time=end_time,
        )

    def _write_to_target(self):
        if self._target:
            is_persistent_vector = self.vector.metadata.name is not None
            if not self._target.path and not is_persistent_vector:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "target path was not specified"
                )
            self._target.set_resource(self.vector)
            size = self._target.write_dataframe(self._result_df)
            if is_persistent_vector:
                target_status = self._target.update_resource_status("ready", size=size)
                logger.info(f"wrote target: {target_status}")
                self.vector.save()

    def _set_indexes(self, df):
        if self._index_columns and not self._drop_indexes:

            if df.index is None or df.index.name is None:
                index_columns_missing = []
                for index in self._index_columns:
                    if index not in df.columns:
                        index_columns_missing.append(index)
                if not index_columns_missing:
                    return df.set_index(self._index_columns)
                else:
                    logger.warn(
                        f"Can't set index, not all index columns found: {index_columns_missing}. "
                        f"It is possible that column was already indexed."
                    )
        return df

    @abc.abstractmethod
    def _generate_vector(
        self,
        entity_rows,
        entity_timestamp_column,
        feature_set_objects,
        feature_set_fields,
        start_time=None,
        end_time=None,
    ):
        raise NotImplementedError("_generate_vector() operation not supported in class")

    def merge(
        self,
        entity_df,
        entity_timestamp_column: str,
        featuresets: list,
        featureset_dfs: list,
    ):
        """join the entities and feature set features into a result dataframe"""
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

    @abc.abstractmethod
    def _asof_join(
        self, entity_df, entity_timestamp_column: str, featureset, featureset_df,
    ):
        raise NotImplementedError("_asof_join() operation not implemented in class")

    @abc.abstractmethod
    def _join(
        self, entity_df, entity_timestamp_column: str, featureset, featureset_df,
    ):
        raise NotImplementedError("_join() operation not implemented in class")

    def get_status(self):
        """return the status of the merge operation (in case its asynchrounious)"""
        if self._result_df is None:
            raise RuntimeError("unexpected status, no result df")
        return "completed"

    def get_df(self, to_pandas=True):
        """return the result as a dataframe (pandas by default)"""
        return self._result_df

    def to_parquet(self, target_path, **kw):
        """return results as parquet file"""
        size = ParquetTarget(path=target_path).write_dataframe(self._result_df, **kw)
        return size

    def to_csv(self, target_path, **kw):
        """return results as csv file"""
        size = CSVTarget(path=target_path).write_dataframe(self._result_df, **kw)
        return size
