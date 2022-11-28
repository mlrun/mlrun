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
from typing import List

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
                        return df.set_index(self._index_columns)
                    elif self.engine == "dask" and len(self._index_columns) == 1:
                        return df.set_index(self._index_columns[0])
                    else:
                        logger.info(
                            "The entities will stay as columns because "
                            "Dask dataframe does not yet support multi-indexes"
                        )
                else:
                    logger.warn(
                        f"Can't set index, not all index columns found: {index_columns_missing}. "
                        f"It is possible that column was already indexed."
                    )
        return self._result_df

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
                keys = [[[], []]] * len(featureset_dfs)
            if all_columns is not None:
                all_columns.pop(0)
            else:
                all_columns = [[]] * len(featureset_dfs)
            entity_timestamp_column = (
                entity_timestamp_column or featureset.spec.timestamp_key
            )

        for i, featureset, featureset_df, lr_key, columns in zip(
            range(len(featuresets)), featuresets, featureset_dfs, keys, all_columns
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

            # unpersist as required by the implementation (e.g. spark) and delete references
            # to dataframe to allow for GC to free up the memory (local, dask)
            self._unpersist_df(featureset_df)
            featureset_dfs[i] = None
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

    def _parse_relations(self, feature_set_objects, feature_set_fields):
        """This method parse all feature set relations to a format such as self._relations if self._relations is None,
        and when check if each relation as entity included"""
        if self._relation == {}:
            for name, _ in feature_set_fields.items():
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

    def create_linked_relation_list(self, feature_set_objects, feature_set_fields):
        relation_linked_lists = []
        for name, _ in feature_set_fields.items():
            feature_set = feature_set_objects[name]
            relations = LinkedList()
            main_node = Node(
                name,
                data={
                    "left_keys": [],
                    "right_keys": [],
                    "save_cols": [],
                    "save_index": [],
                },
            )
            relations.add_first(main_node)

            # build list of relation and index match for each feature set
            for name_in, _ in feature_set_fields.items():
                if name_in == name:
                    continue
                feature_set_in_entity_list = feature_set_objects[name_in].spec.entities
                feature_set_in_entity_list_names = [*feature_set_in_entity_list.keys()]
                entity_relation_list = [*feature_set.spec.relations.values()]
                col_relation_list = [*feature_set.spec.relations.keys()]
                curr_col_relation_list = []
                relation_wise = True
                for ent in feature_set_in_entity_list:
                    # checking if feature_set have relation with feature_set_in
                    if ent not in entity_relation_list:
                        relation_wise = False
                        break
                    curr_col_relation_list.append(
                        col_relation_list[entity_relation_list.index(ent)]
                    )
                if relation_wise:
                    # add to the link list feature set according to the defined relation
                    relations.add_last(
                        Node(
                            name_in,
                            data={
                                "left_keys": curr_col_relation_list,
                                "right_keys": feature_set_in_entity_list_names,
                                "save_cols": [],
                                "save_index": [],
                            },
                        )
                    )
                    main_node.data["save_cols"].append(*curr_col_relation_list)
                elif sorted(feature_set_in_entity_list_names) == sorted(
                    feature_set.spec.entities.keys()
                ):
                    # add to the link list feature set according to indexes match
                    keys = feature_set_in_entity_list_names
                    relations.add_last(
                        Node(
                            name_in,
                            data={
                                "left_keys": keys,
                                "right_keys": keys,
                                "save_cols": [],
                                "save_index": keys,
                            },
                        )
                    )
                    main_node.data["save_index"] = keys

            relation_linked_lists.append(relations)

        # concat all the link lists to one, for the merging process
        link_list_iter = relation_linked_lists.__iter__()
        return_relation = link_list_iter.__next__()
        for relation_list in link_list_iter:
            return_relation.concat(relation_list, ["save_cols"])
        if return_relation.len != len(feature_set_objects):
            raise mlrun.errors.MLRunRuntimeError("Failed to merge")

        return return_relation


class Node:
    def __init__(self, name: str, data=None):
        self.name = name
        self.data = data
        self.next = None

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name


class LinkedList:
    def __init__(self):
        self.head = None
        self.len = 0

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

    def add_first(self, node: Node):
        node.next = self.head
        self.head = node
        self.len += 1
        if self.head is None:
            self.head = node

    def add_last(self, node: Node):
        if self.head is None:
            self.head = node
            return
        for current_node in self:
            pass
        current_node.next = node
        self.len += 1

    def add_after(self, target_node_name: str, new_node: Node):
        if self.head is None:
            raise Exception("List is empty")

        for node in self:
            if node.name == target_node_name:
                new_node.next = node.next
                node.next = new_node
                self.len += 1
                return

        raise Exception("Node with data '%s' not found" % target_node_name)

    def find_node(self, target_node_name: str):
        if self.head is None:
            return None

        for node in self:
            if node.name == target_node_name:
                return node

    def concat(self, other, data_attributes: List[str]):
        other_iter = other.__iter__()
        other_head = other_iter.__next__()
        node = self.find_node(other_head.name)
        for atr in data_attributes:
            node.data[atr] = other_head.data[atr]
        if node is None:
            raise mlrun.errors.MLRunRuntimeError(
                f"Can't join those {other_head.name} and {self.head.name} feature sets"
            )
        for other_node in other_iter:
            if self.find_node(other_node.name) is None:
                self.add_after(node.name, other_node)
                node = other_node
