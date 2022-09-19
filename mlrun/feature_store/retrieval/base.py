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
import abc

import mlrun
from mlrun.datastore.targets import CSVTarget, ParquetTarget

from ...utils import logger


class BaseMerger(abc.ABC):
    """abstract feature merger class"""

    def __init__(self, vector, **engine_args):
        self._relation = dict()
        self._join_type = "inner"
        self.vector = vector

        self._result_df = None
        self._drop_columns = []
        self._index_columns = []
        self._drop_indexes = True
        self._target = None
        self._alias = dict()

    def _append_drop_column(self, key):
        if key and key not in self._drop_columns:
            self._drop_columns.append(key)

    def _append_index(self, key):
        if key:
            if key not in self._index_columns:
                self._index_columns.append(key)
            if self._drop_indexes:
                self._append_drop_column(key)

    def _update_alias(self, key: str = None, val: str = None, dictionary: dict = None):
        if dictionary is not None:
            # adding dictionary to alias
            self._alias.update(dictionary)
        elif val in self._alias.values():
            # changing alias key
            old_key = [key for key, v in self._alias.items() if v == val][0]
            self._alias[key] = self._alias.pop(old_key)
        else:
            self._alias[key] = val

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
        query=None,
        join_type="inner",
        relations=None,
    ):
        self._target = target
        self._join_type = join_type
        self._relation = relations or {}

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
            query=query,
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
        query=None,
    ):
        raise NotImplementedError("_generate_vector() operation not supported in class")

    def merge(
        self,
        entity_df,
        entity_timestamp_column: str,
        featuresets: list,
        featureset_dfs: list,
        keys: list = None,
        all_columns: list = None,
    ):
        """join the entities and feature set features into a result dataframe"""
        merged_df = entity_df
        if entity_df is None and featureset_dfs:
            merged_df = featureset_dfs.pop(0)
            featureset = featuresets.pop(0)
            if keys is not None:
                keys.pop(0)
            else:
                keys = [[[], []]] * len(featureset_dfs)
            if all_columns is not None:
                all_columns.pop(0)
            else:
                all_columns = [[]] * len(featureset_dfs)
            entity_timestamp_column = (
                entity_timestamp_column or featureset.spec.timestamp_key
            )

        for featureset, featureset_df, lr_key, columns in zip(
            featuresets, featureset_dfs, keys, all_columns
        ):
            if featureset.spec.timestamp_key:
                merge_func = self._asof_join
            else:
                merge_func = self._join

            merged_df = merge_func(
                merged_df,
                entity_timestamp_column,
                featureset,
                featureset_df,
                lr_key[0],
                lr_key[1],
                columns,
            )

        self._result_df = merged_df

    @abc.abstractmethod
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
        raise NotImplementedError("_asof_join() operation not implemented in class")

    @abc.abstractmethod
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

    def _parse_relations(self, feature_set_objects, feature_set_fields):
        if self._relation == {}:
            for name, columns in feature_set_fields.items():
                feature_set = feature_set_objects[name]
                fs_relations = feature_set.spec.relations
                for other_fs, relation in fs_relations.items():
                    if (
                        f"{name}:{other_fs}" not in self._relation
                        or f"{other_fs}:{name}" not in self._relation
                    ):
                        self._relation[f"{name}:{other_fs}"] = relation

        # check all relations are included entities
        for relation_name, relation in self._relation.items():
            first_fs, second_fs = relation_name.split(":")
            first_entities, second_entities = (
                feature_set_objects[first_fs].spec.entities.keys(),
                feature_set_objects[second_fs].spec.entities.keys(),
            )
            for key, val in relation.items():
                if key not in first_entities and val not in second_entities:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"Relation have to include entities therefore "
                        f"the relation {key}:{val} between {first_fs} to "
                        f"{second_fs} is not valid"
                    )
