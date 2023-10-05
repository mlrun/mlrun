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
import abc
import typing
from datetime import datetime

import pandas as pd

import mlrun
from mlrun.datastore.targets import CSVTarget, ParquetTarget
from mlrun.feature_store.feature_set import FeatureSet
from mlrun.feature_store.feature_vector import JoinGraph

from ...utils import logger, str_to_timestamp
from ..feature_vector import OfflineVectorResponse


class BaseMerger(abc.ABC):
    """abstract feature merger class"""

    # In order to be an online merger, the merger should implement `init_online_vector_service` function.
    support_online = False

    # In order to be an offline merger, the merger should implement
    # `_order_by`, `_filter`, `_drop_columns_from_result`, `_rename_columns_and_select`, `_get_engine_df` functions.
    support_offline = False
    engine = None

    def __init__(self, vector, **engine_args):
        self._relation = dict()
        self._join_type = "inner"
        self._default_join_type = "default_join"
        self.vector = vector

        self._result_df = None
        self._drop_columns = []
        self._index_columns = []
        self._drop_indexes = True
        self._target = None
        self._alias = dict()
        self._origin_alias = dict()
        self._entity_rows_node_name = "__mlrun__$entity_rows$"

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
        timestamp_for_filtering=None,
        with_indexes=None,
        update_stats=None,
        query=None,
        order_by=None,
    ):
        self._target = target

        # calculate the index columns and columns we need to drop
        self._drop_columns = drop_columns or self._drop_columns
        if self.vector.spec.with_indexes or with_indexes:
            self._drop_indexes = False

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

        if self._drop_indexes and entity_timestamp_column:
            self._append_drop_column(entity_timestamp_column)

        for feature_set in feature_set_objects.values():
            if self._drop_indexes:
                self._append_drop_column(feature_set.spec.timestamp_key)
            for key in feature_set.spec.entities.keys():
                self._append_index(key)

        start_time = str_to_timestamp(start_time)
        end_time = str_to_timestamp(end_time)
        if start_time and not end_time:
            # if end_time is not specified set it to now()
            end_time = pd.Timestamp.now()

        return self._generate_offline_vector(
            entity_rows,
            entity_timestamp_column,
            feature_set_objects=feature_set_objects,
            feature_set_fields=feature_set_fields,
            start_time=start_time,
            end_time=end_time,
            timestamp_for_filtering=timestamp_for_filtering,
            query=query,
            order_by=order_by,
        )

    def _write_to_offline_target(self, timestamp_key=None):
        save_vector = False
        if not self._drop_indexes and timestamp_key not in self._drop_columns:
            self.vector.status.timestamp_key = timestamp_key
            save_vector = True
        if self._target:
            is_persistent_vector = self.vector.metadata.name is not None
            if not self._target.path and not is_persistent_vector:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "target path was not specified"
                )
            self._target.set_resource(self.vector)
            size = self._target.write_dataframe(
                self._result_df, timestamp_key=self.vector.status.timestamp_key
            )
            if is_persistent_vector:
                target_status = self._target.update_resource_status("ready", size=size)
                logger.info(f"wrote target: {target_status}")
            save_vector = True
        if save_vector:
            self.vector.save()

    def _set_indexes(self, df):
        if self._index_columns and not self._drop_indexes:
            if df.index is None or df.index.name is None:
                index_columns_missing = []
                for index in self._index_columns:
                    if index not in df.columns:
                        index_columns_missing.append(index)
                if not index_columns_missing:
                    df.set_index(self._index_columns, inplace=True)
                else:
                    logger.warn(
                        f"Can't set index, not all index columns found: {index_columns_missing}. "
                        f"It is possible that column was already indexed."
                    )
        else:
            df.reset_index(drop=True, inplace=True)

    def _generate_offline_vector(
        self,
        entity_rows,
        entity_timestamp_column,
        feature_set_objects,
        feature_set_fields,
        start_time=None,
        end_time=None,
        timestamp_for_filtering=None,
        query=None,
        order_by=None,
    ):
        self._create_engine_env()

        feature_sets = []
        dfs = []
        keys = (
            []
        )  # the struct of key is [[[],[]], ..] So that each record indicates which way the corresponding
        # featureset is connected to the previous one, and within each record the left keys are indicated in index 0
        # and the right keys in index 1, this keys will be the keys that will be used in this join
        join_types = []

        if entity_rows is not None:
            if entity_rows.index.names[0]:
                entity_rows = entity_rows.reset_index()
            entity_rows_keys = list(entity_rows.columns)
        else:
            entity_rows_keys = None
        join_graph = self._get_graph(
            feature_set_objects, feature_set_fields, entity_rows_keys
        )
        if entity_rows_keys:
            entity_rows = self._convert_entity_rows_to_engine_df(entity_rows)
            dfs.append(entity_rows)
            keys.append([[], []])
            feature_sets.append(None)
            join_types.append(None)

        filtered = False
        for step in join_graph.steps:
            name = step.right_feature_set_name
            feature_set = feature_set_objects[name]
            saved_columns_for_relation = list(
                self.vector.get_feature_set_relations(feature_set).keys()
            )
            feature_sets.append(feature_set)
            columns = feature_set_fields[name]
            self._origin_alias.update({name: alias for name, alias in columns})
            column_names = [name for name, _ in columns]

            for column in saved_columns_for_relation:
                if column not in column_names:
                    column_names.append(column)
                    if column not in self._index_columns:
                        self._append_drop_column(column)

            if isinstance(timestamp_for_filtering, dict):
                time_column = timestamp_for_filtering.get(
                    name, feature_set.spec.timestamp_key
                )
            elif isinstance(timestamp_for_filtering, str):
                time_column = timestamp_for_filtering
            else:
                time_column = feature_set.spec.timestamp_key

            if time_column != feature_set.spec.timestamp_key and time_column not in [
                feature.name for feature in feature_set.spec.features
            ]:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Feature set `{name}` "
                    f"does not have a column named `{time_column}` to filter on."
                )

            if self._drop_indexes:
                self._append_drop_column(time_column)
            if (start_time or end_time) and time_column:
                filtered = True

            df = self._get_engine_df(
                feature_set,
                name,
                column_names,
                start_time if time_column else None,
                end_time if time_column else None,
                time_column,
            )

            fs_entities_and_timestamp = list(feature_set.spec.entities.keys())
            column_names += fs_entities_and_timestamp
            saved_columns_for_relation += fs_entities_and_timestamp
            if feature_set.spec.timestamp_key:
                column_names.append(feature_set.spec.timestamp_key)
                saved_columns_for_relation.append(feature_set.spec.timestamp_key)
                fs_entities_and_timestamp.append(feature_set.spec.timestamp_key)

            # rename columns to be unique for each feature set and select if needed
            rename_col_dict = {
                column: f"{column}_{name}"
                for column in column_names
                if column not in saved_columns_for_relation
            }
            df_temp = self._rename_columns_and_select(
                df,
                rename_col_dict,
                columns=list(set(column_names + fs_entities_and_timestamp)),
            )

            if df_temp is not None:
                df = df_temp
                del df_temp

            dfs.append(df)
            del df

            keys.append([step.left_keys, step.right_keys])
            join_types.append([step.join_type, step.asof_join])

            # update alias according to the unique column name
            new_columns = []
            if not self._drop_indexes:
                new_columns.extend([(ind, ind) for ind in fs_entities_and_timestamp])
            for column, alias in columns:
                if column in rename_col_dict:
                    new_columns.append((rename_col_dict[column], alias or column))
                else:
                    new_columns.append((column, alias))
            self._update_alias(dictionary={name: alias for name, alias in new_columns})

        # None of the feature sets was filtered as required
        if not filtered and (start_time or end_time):
            raise mlrun.errors.MLRunRuntimeError(
                "start_time and end_time can only be provided in conjunction with "
                "a timestamp column, or when the at least one feature_set has a timestamp key"
            )
        # join the feature data frames
        result_timestamp = self.merge(
            entity_timestamp_column=entity_timestamp_column,
            featuresets=feature_sets,
            featureset_dfs=dfs,
            keys=keys,
            join_types=join_types,
        )

        all_columns = None
        if not self._drop_indexes and result_timestamp:
            if result_timestamp not in self._alias.values():
                self._update_alias(key=result_timestamp, val=result_timestamp)
            all_columns = list(self._alias.keys())

        df_temp = self._rename_columns_and_select(
            self._result_df, self._alias, columns=all_columns
        )
        if df_temp is not None:
            self._result_df = df_temp
            del df_temp

        df_temp = self._drop_columns_from_result()
        if df_temp is not None:
            self._result_df = df_temp
            del df_temp

        if self.vector.status.label_column:
            self._result_df = self._result_df.dropna(
                subset=[self.vector.status.label_column]
            )
        # filter joined data frame by the query param
        if query:
            self._filter(query)

        if order_by:
            if isinstance(order_by, str):
                order_by = [order_by]
            order_by_active = [
                order_col
                if order_col in self._result_df.columns
                else self._origin_alias.get(order_col, None)
                for order_col in order_by
            ]
            if None in order_by_active:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"Result dataframe contains {self._result_df.columns} "
                    f"columns and can't order by {order_by}"
                )
            self._order_by(order_by_active)

        self._write_to_offline_target(timestamp_key=result_timestamp)
        return OfflineVectorResponse(self)

    def init_online_vector_service(
        self, entity_keys, fixed_window_type, update_stats=False
    ):
        """
        initialize the `OnlineVectorService`

        :param entity_keys:         list of the feature_vector indexes.
        :param fixed_window_type:   determines how to query the fixed window values which were previously
                                    inserted by ingest
        :param update_stats:        update features statistics from the requested feature sets on the vector.
                                    Default: False.

        :return:                    `OnlineVectorService`
        """
        raise NotImplementedError

    def _unpersist_df(self, df):
        pass

    def merge(
        self,
        entity_timestamp_column: str,
        featuresets: list,
        featureset_dfs: list,
        keys: list = None,
        join_types: list = None,
    ):
        """join the entities and feature set features into a result dataframe"""

        merged_df = featureset_dfs.pop(0)
        featureset = featuresets.pop(0)
        keys.pop(0)
        join_types.pop(0)

        if not entity_timestamp_column and featureset:
            entity_timestamp_column = featureset.spec.timestamp_key

        for featureset, featureset_df, lr_key, join_type in zip(
            featuresets, featureset_dfs, keys, join_types
        ):
            join_type, as_of = join_type
            if (
                featureset.spec.timestamp_key
                and entity_timestamp_column
                and join_type == self._default_join_type
            ):
                merge_func = self._asof_join
            elif join_type == self._default_join_type:
                merge_func = self._join
            elif join_type != self._default_join_type and not as_of:
                self._join_type = join_type
                merge_func = self._join
            else:
                self._join_type = join_type
                merge_func = self._asof_join

            merged_df = merge_func(
                merged_df,
                entity_timestamp_column,
                featureset.metadata.name,
                featureset.spec.timestamp_key,
                featureset_df,
                lr_key[0],
                lr_key[1],
            )
            entity_timestamp_column = (
                entity_timestamp_column or featureset.spec.timestamp_key
            )

            # unpersist as required by the implementation (e.g. spark) and delete references
            # to dataframe to allow for GC to free up the memory (local, dask)
            self._unpersist_df(featureset_df)
            del featureset_df

        self._result_df = merged_df
        return entity_timestamp_column

    def _asof_join(
        self,
        entity_df,
        entity_timestamp_column: str,
        featureset_name: str,
        featureset_timstamp: str,
        featureset_df,
        left_keys: list,
        right_keys: list,
    ):
        raise NotImplementedError("_asof_join() operation not implemented in class")

    def _join(
        self,
        entity_df,
        entity_timestamp_column: str,
        featureset_name: str,
        featureset_timestamp: str,
        featureset_df,
        left_keys: list,
        right_keys: list,
    ):
        raise NotImplementedError("_join() operation not implemented in class")

    def get_status(self):
        """return the status of the merge operation (in case its asynchrounious)"""
        if self._result_df is None:
            raise RuntimeError("unexpected status, no result df")
        return "completed"

    def get_df(self, to_pandas=True):
        """return the result as a dataframe (pandas by default)"""
        self._set_indexes(self._result_df)
        return self._result_df

    def to_parquet(self, target_path, **kw):
        """return results as parquet file"""
        size = ParquetTarget(path=target_path).write_dataframe(self._result_df, **kw)
        return size

    def to_csv(self, target_path, **kw):
        """return results as csv file"""
        size = CSVTarget(path=target_path).write_dataframe(self._result_df, **kw)
        return size

    def _get_graph(
        self, feature_set_objects, feature_set_fields, entity_rows_keys=None
    ):
        join_graph = self.vector.spec.join_graph
        if not join_graph:
            fs_link_list = self._create_linked_relation_list(
                feature_set_objects, feature_set_fields, entity_rows_keys
            )
            join_graph = None
            for i, node in enumerate(fs_link_list):
                if node.name != self._entity_rows_node_name and join_graph is None:
                    join_graph = JoinGraph(first_feature_set=node.name)
                elif node.name == self._entity_rows_node_name:
                    continue
                else:
                    join_graph.inner(other_operand=node.name)

                last_step = join_graph.steps[-1]
                last_step.join_type = self._default_join_type
                last_step.left_keys = node.left_keys
                last_step.right_keys = node.right_keys
        else:
            join_graph._init_all_join_keys(feature_set_objects, self.vector)
        return join_graph

    class _Node:
        def __init__(
            self,
            name: str,
            order: int,
            left_keys: typing.List[str] = None,
            right_keys: typing.List[str] = None,
        ):
            self.name = name
            self.left_keys = left_keys if left_keys is not None else []
            self.right_keys = right_keys if right_keys is not None else []
            # order of this feature_set in the original list
            self.order = order
            self.next = None

        def __repr__(self):
            return self.name

        def __eq__(self, other):
            return self.name == other.name

        def __copy__(self):
            return BaseMerger._Node(
                self.name, self.order, self.left_keys, self.right_keys
            )

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

        def __copy__(self):
            ll = BaseMerger._LinkedList()
            prev_node = None
            for node in self:
                new_node = node.__copy__()
                if ll.head is None:
                    ll.head = new_node
                else:
                    prev_node.next = new_node
                prev_node = new_node
            ll.len = self.len
            return ll

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
            while node:
                self.len += 1
                node = node.next

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
            for other_node in other_iter:
                if self.find_node(other_node.name) is None:
                    while node is not None and other_node.order > node.order:
                        node = node.next
                    if node:
                        self.add_after(node, other_node)
                    else:
                        self.add_last(other_node)
                    node = other_node

    def _create_linked_relation_list(
        self, feature_set_objects, feature_set_fields, entity_rows_keys=None
    ):
        feature_set_names = list(feature_set_fields.keys())
        if len(feature_set_names) == 1 and not entity_rows_keys:
            return BaseMerger._LinkedList(
                head=BaseMerger._Node(
                    name=feature_set_names[0],
                    order=0,
                )
            )
        relation_linked_lists = []
        feature_set_entity_list_dict = {
            name: feature_set_objects[name].spec.entities for name in feature_set_names
        }
        relation_val_list = {
            name: list(
                self.vector.get_feature_set_relations(
                    feature_set_objects[name]
                ).values()
            )
            for name in feature_set_names
        }
        relation_key_list = {
            name: list(
                self.vector.get_feature_set_relations(feature_set_objects[name]).keys()
            )
            for name in feature_set_names
        }

        def _create_relation(name: str, order):
            relations = BaseMerger._LinkedList()
            main_node = BaseMerger._Node(
                name,
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
            entity_relation_list = relation_val_list[name_head]
            col_relation_list = relation_key_list[name_head]
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

            if all(
                curr_col_relation_list
            ):  # checking if feature_set have relation with feature_set_in
                # add to the link list feature set according to the defined relation
                linked_list_relation.add_last(
                    BaseMerger._Node(
                        fs_name_in,
                        left_keys=curr_col_relation_list,
                        right_keys=feature_set_in_entity_list_names,
                        order=name_in_order,
                    )
                )
            elif name_in_order > head_order and sorted(
                feature_set_in_entity_list_names
            ) == sorted(feature_set_entity_list_dict[name_head].keys()):
                # add to the link list feature set according to indexes match
                keys = feature_set_in_entity_list_names
                linked_list_relation.add_last(
                    BaseMerger._Node(
                        fs_name_in,
                        left_keys=keys,
                        right_keys=keys,
                        order=name_in_order,
                    )
                )
            return linked_list_relation

        def _build_entity_rows_relation(entity_rows_relation, fs_name, fs_order):
            feature_set_entity_list = feature_set_entity_list_dict[fs_name]
            feature_set_entity_list_names = list(feature_set_entity_list.keys())

            if all([ent in entity_rows_keys for ent in feature_set_entity_list_names]):
                # add to the link list feature set according to indexes match,
                # only if all entities in the feature set exist in the entity rows
                keys = feature_set_entity_list_names
                entity_rows_relation.add_last(
                    BaseMerger._Node(
                        fs_name,
                        left_keys=keys,
                        right_keys=keys,
                        order=fs_order,
                    )
                )

        if entity_rows_keys is not None:
            entity_rows_linked_relation = _create_relation(
                self._entity_rows_node_name, -1
            )
            relation_linked_lists.append(entity_rows_linked_relation)
            linked_list_len_goal = len(feature_set_objects) + 1
        else:
            entity_rows_linked_relation = None
            linked_list_len_goal = len(feature_set_objects)

        for i, name in enumerate(feature_set_names):
            linked_relation = _create_relation(name, i)
            if entity_rows_linked_relation is not None:
                _build_entity_rows_relation(entity_rows_linked_relation, name, i)
            for j, name_in in enumerate(feature_set_names):
                if name != name_in:
                    linked_relation = _build_relation(name_in, j, linked_relation, i)
            relation_linked_lists.append(linked_relation)

        # concat all the link lists to one, for the merging process
        for i in range(len(relation_linked_lists)):
            return_relation = relation_linked_lists[i].__copy__()
            for relation_list in relation_linked_lists:
                return_relation.concat(relation_list)
            if return_relation.len == linked_list_len_goal:
                return return_relation

        raise mlrun.errors.MLRunRuntimeError("Failed to merge")

    def get_default_image(cls, kind):
        return mlrun.mlconf.feature_store.default_job_image

    def _reset_index(self, _result_df):
        raise NotImplementedError

    def _create_engine_env(self):
        """
        initialize engine env if needed
        """
        raise NotImplementedError

    def _get_engine_df(
        self,
        feature_set: FeatureSet,
        feature_set_name: typing.List[str],
        column_names: typing.List[str] = None,
        start_time: typing.Union[str, datetime] = None,
        end_time: typing.Union[str, datetime] = None,
        time_column: typing.Optional[str] = None,
    ):
        """
        Return the feature_set data frame according to the args

        :param feature_set:             current feature_set to extract from the data frame
        :param feature_set_name:        the name of the current feature_set
        :param column_names:            list of columns to select (if not all)
        :param start_time:              filter by start time
        :param end_time:                filter by end time
        :param time_column:             specify the time column name to filter on

        :return: Data frame of the current engine
        """
        raise NotImplementedError

    def _rename_columns_and_select(
        self,
        df,
        rename_col_dict: typing.Dict[str, str],
        columns: typing.List[str] = None,
    ):
        """
        rename the columns of the df according to rename_col_dict, and select only `columns` if it is not none

        :param df:              the data frame to change
        :param rename_col_dict: the renaming dictionary - {<current_column_name>: <new_column_name>, ...}
        :param columns:         list of columns to select (if not all)

        :return: the data frame after the transformation or None if the transformation were preformed inplace
        """
        raise NotImplementedError

    def _drop_columns_from_result(self):
        """
        drop `self._drop_columns` from `self._result_df`
        """
        raise NotImplementedError

    def _filter(self, query: str):
        """
        filter `self._result_df` by `query`

        :param query: The query string used to filter rows
        """
        raise NotImplementedError

    def _order_by(self, order_by_active: typing.List[str]):
        """
        Order by `order_by_active` along all axis.

        :param order_by_active: list of names to sort by.
        """
        raise NotImplementedError

    def _convert_entity_rows_to_engine_df(self, entity_rows):
        raise NotImplementedError
