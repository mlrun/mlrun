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

    engine = None

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
    ):
        self._target = target
        self._join_type = join_type

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
                    if self.engine == "local" or self.engine == "spark":
                        df.set_index(self._index_columns, inplace=True)
                    elif self.engine == "dask":
                        if len(self._index_columns) == 1:
                            return df.set_index(self._index_columns[0])
                        elif len(self._index_columns) != 1:
                            return self._reset_index(self._result_df)
                        else:
                            logger.info(
                                "The entities will stay as columns because "
                                "Dask dataframe does not yet support multi-indexes"
                            )
                            return self._result_df
                else:
                    logger.warn(
                        f"Can't set index, not all index columns found: {index_columns_missing}. "
                        f"It is possible that column was already indexed."
                    )
            else:
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

    def _unpersist_df(self, df):
        pass

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
                # keys can be multiple keys on each side of the join
                keys = [[[], []]] * len(featureset_dfs)
            if all_columns is not None:
                all_columns.pop(0)
            else:
                all_columns = [[]] * len(featureset_dfs)
            entity_timestamp_column = (
                entity_timestamp_column or featureset.spec.timestamp_key
            )
        elif entity_df is not None and featureset_dfs:
            # when `entity_rows` passed to `get_offline_features`
            # keys[0] mention the way that `entity_rows`  joins to the first `featureset`
            # and it can join only by the entities of the first `featureset`
            keys[0][0] = keys[0][1] = list(featuresets[0].spec.entities.keys())

        for featureset, featureset_df, lr_key, columns in zip(
            featuresets, featureset_dfs, keys, all_columns
        ):
            if featureset.spec.timestamp_key:
                merge_func = self._asof_join
                if self._join_type != "inner":
                    logger.warn(
                        "Merge all the features with as_of_join and don't "
                        "take into account the join_type that was given"
                    )
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

            # unpersist as required by the implementation (e.g. spark) and delete references
            # to dataframe to allow for GC to free up the memory (local, dask)
            self._unpersist_df(featureset_df)
            del featureset_df

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

    class _Node:
        def __init__(self, name: str, order: int, data=None):
            self.name = name
            self.data = data
            # order of this feature_set in the original list
            self.order = order
            self.next = None

        def __repr__(self):
            return self.name

        def __eq__(self, other):
            return self.name == other.name

    class _LinkedList:
        def __init__(self, head=None):
            self.head = head
            self.len = 1 if head is not None else 0

        def __repr__(self):
            node = self.head
            nodes = []
            while node is not None:
                nodes.append(node.name)
                node = node.next
            nodes.append("None")
            return " -> ".join(nodes)

        def __iter__(self):
            node = self.head
            while node is not None:
                yield node
                node = node.next

        def add_first(self, node):
            node.next = self.head
            self.head = node
            self.len += 1

        def add_last(self, node):
            if self.head is None:
                self.head = node
                return
            for current_node in self:
                pass
            current_node.next = node
            self.len += 1

        def add_after(self, target_node, new_node):
            new_node.next = target_node.next
            target_node.next = new_node
            self.len += 1

        def find_node(self, target_node_name: str):
            if self.head is None:
                return None

            for node in self:
                if node.name == target_node_name:
                    return node

        def concat(self, other):
            other_iter = iter(other)
            other_head = next(other_iter)
            node = self.find_node(other_head.name)
            if node is None:
                return
            node.data["save_cols"] += other_head.data["save_cols"]
            for other_node in other_iter:
                if self.find_node(other_node.name) is None:
                    while node is not None and other_node.order > node.order:
                        node = node.next
                    if node:
                        self.add_after(node, other_node)
                    else:
                        self.add_last(other_node)
                    node = other_node

    @staticmethod
    def _create_linked_relation_list(feature_set_objects, feature_set_fields):
        feature_set_names = list(feature_set_fields.keys())
        if len(feature_set_names) == 1:
            return BaseMerger._LinkedList(
                head=BaseMerger._Node(
                    name=feature_set_names[0],
                    order=0,
                    data={
                        "left_keys": [],
                        "right_keys": [],
                        "save_cols": [],
                        "save_index": [],
                    },
                )
            )
        relation_linked_lists = []
        feature_set_entity_list_dict = {
            name: feature_set_objects[name].spec.entities for name in feature_set_names
        }
        entity_relation_val_list = {
            name: list(feature_set_objects[name].spec.relations.values())
            for name in feature_set_names
        }
        entity_relation_key_list = {
            name: list(feature_set_objects[name].spec.relations.keys())
            for name in feature_set_names
        }

        def _create_relation(name: str, order):
            relations = BaseMerger._LinkedList()
            main_node = BaseMerger._Node(
                name,
                data={
                    "left_keys": [],
                    "right_keys": [],
                    "save_cols": [],
                    "save_index": [],
                },
                order=order,
            )
            relations.add_first(main_node)
            return relations

        def _build_relation(
            fs_name_in: str, name_in_order, linked_list_relation, head_order
        ):
            name_head = linked_list_relation.head.name
            feature_set_in_entity_list = feature_set_entity_list_dict[fs_name_in]
            feature_set_in_entity_list_names = list(feature_set_in_entity_list.keys())
            entity_relation_list = entity_relation_val_list[name_head]
            col_relation_list = entity_relation_key_list[name_head]
            curr_col_relation_list = list(
                map(
                    lambda ent: (
                        col_relation_list[entity_relation_list.index(ent)]
                        if ent in entity_relation_list
                        else False
                    ),
                    feature_set_in_entity_list,
                )
            )

            # checking if feature_set have relation with feature_set_in
            relation_wise = all(curr_col_relation_list)

            if relation_wise:
                # add to the link list feature set according to the defined relation
                linked_list_relation.add_last(
                    BaseMerger._Node(
                        fs_name_in,
                        data={
                            "left_keys": curr_col_relation_list,
                            "right_keys": feature_set_in_entity_list_names,
                            "save_cols": [],
                            "save_index": [],
                        },
                        order=name_in_order,
                    )
                )
                linked_list_relation.head.data["save_cols"].append(
                    *curr_col_relation_list
                )
            elif name_in_order > head_order and sorted(
                feature_set_in_entity_list_names
            ) == sorted(feature_set_entity_list_dict[name_head].keys()):
                # add to the link list feature set according to indexes match
                keys = feature_set_in_entity_list_names
                linked_list_relation.add_last(
                    BaseMerger._Node(
                        fs_name_in,
                        data={
                            "left_keys": keys,
                            "right_keys": keys,
                            "save_cols": [],
                            "save_index": keys,
                        },
                        order=name_in_order,
                    )
                )
                linked_list_relation.head.data["save_index"] = keys
            return linked_list_relation

        for i, name in enumerate(feature_set_names):
            linked_relation = _create_relation(name, i)
            for j, name_in in enumerate(feature_set_names):
                if name != name_in:
                    linked_relation = _build_relation(name_in, j, linked_relation, i)
            relation_linked_lists.append(linked_relation)

        # concat all the link lists to one, for the merging process
        link_list_iter = iter(relation_linked_lists)
        return_relation = next(link_list_iter)
        for relation_list in link_list_iter:
            return_relation.concat(relation_list)
        if return_relation.len != len(feature_set_objects):
            raise mlrun.errors.MLRunRuntimeError("Failed to merge")

        return return_relation

    @classmethod
    def get_default_image(cls, kind):
        return mlrun.mlconf.feature_store.default_job_image

    def _reset_index(self, _result_df):
        raise NotImplementedError
