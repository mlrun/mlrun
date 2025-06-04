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
import typing
from copy import copy
from typing import Union

import numpy as np

import mlrun
from mlrun.feature_store import FeatureSet

from ..model import ModelObj, ObjectList

# Forward reference for type annotations


class _JoinStep(ModelObj):
    def __init__(
        self,
        name: typing.Optional[str] = None,
        left_step_name: typing.Optional[str] = None,
        right_step_name: typing.Optional[str] = None,
        left_feature_set_names: typing.Optional[Union[str, list[str]]] = None,
        right_feature_set_name: typing.Optional[str] = None,
        join_type: str = "inner",
        asof_join: bool = False,
    ):
        self.name = name
        self.left_step_name = left_step_name
        self.right_step_name = right_step_name
        self.left_feature_set_names = (
            left_feature_set_names
            if left_feature_set_names is None
            or isinstance(left_feature_set_names, list)
            else [left_feature_set_names]
        )
        self.right_feature_set_name = right_feature_set_name
        self.join_type = join_type
        self.asof_join = asof_join

        self.left_keys = []
        self.right_keys = []

    def init_join_keys(
        self,
        feature_set_objects: ObjectList,
        vector,
        entity_rows_keys: typing.Optional[list[str]] = None,
    ):
        if feature_set_objects[self.right_feature_set_name].is_connectable_to_df(
            entity_rows_keys
        ):
            self.left_keys, self.right_keys = [
                list(
                    feature_set_objects[
                        self.right_feature_set_name
                    ].spec.entities.keys()
                )
            ] * 2

        if (
            self.join_type == JoinGraph.first_join_type
            or not self.left_feature_set_names
        ):
            self.join_type = (
                "inner"
                if self.join_type == JoinGraph.first_join_type
                else self.join_type
            )
            return

        for left_fset in self.left_feature_set_names:
            current_left_keys = feature_set_objects[left_fset].extract_relation_keys(
                feature_set_objects[self.right_feature_set_name],
                vector.get_feature_set_relations(feature_set_objects[left_fset]),
            )
            current_right_keys = list(
                feature_set_objects[self.right_feature_set_name].spec.entities.keys()
            )
            for i in range(len(current_left_keys)):
                if (
                    current_left_keys[i] not in self.left_keys
                    and current_right_keys[i] not in self.right_keys
                ):
                    self.left_keys.append(current_left_keys[i])
                    self.right_keys.append(current_right_keys[i])

        if not self.left_keys:
            raise mlrun.errors.MLRunRuntimeError(
                f"{self.name} can't be preform due to undefined relation between "
                f"{self.left_feature_set_names} to {self.right_feature_set_name}"
            )


class JoinGraph(ModelObj):
    """
    A class that represents a graph of data joins between feature sets
    """

    default_graph_name = "$__join_graph_fv__$"
    first_join_type = "first"
    _dict_fields = ["name", "first_feature_set", "steps"]

    def __init__(
        self,
        name: typing.Optional[str] = None,
        first_feature_set: Union[str, FeatureSet] = None,
    ):
        """
        JoinGraph is a class that represents a graph of data joins between feature sets. It allows users to define
        data joins step by step, specifying the join type for each step. The graph can be used to build a sequence of
        joins that will be executed in order, allowing the creation of complex join operations between feature sets.


        Example:
        # Create a new JoinGraph and add steps for joining feature sets.
        join_graph = JoinGraph(name="my_join_graph", first_feature_set="featureset1")
        join_graph.inner("featureset2")
        join_graph.left("featureset3", asof_join=True)


        :param name:                (str, optional) The name of the join graph. If not provided,
                                    a default name will be used.
        :param first_feature_set:   (str or FeatureSet, optional) The first feature set to join. It can be
                                    specified either as a string representing the name of the feature set or as a
                                    FeatureSet object.
        """
        self.name = name or self.default_graph_name
        self._steps: ObjectList = None
        self._feature_sets = None
        if first_feature_set:
            self._start(first_feature_set)

    def inner(self, other_operand: typing.Union[str, FeatureSet]):
        """
        Specifies an inner join with the given feature set

        :param other_operand:       (str or FeatureSet) The name of the feature set or a FeatureSet object to join with.

        :return:                    JoinGraph: The updated JoinGraph object with the specified inner join.
        """
        return self._join_operands(other_operand, "inner")

    def outer(self, other_operand: typing.Union[str, FeatureSet]):
        """
        Specifies an outer join with the given feature set

        :param other_operand:       (str or FeatureSet) The name of the feature set or a FeatureSet object to join with.
        :return:                    JoinGraph: The updated JoinGraph object with the specified outer join.
        """
        return self._join_operands(other_operand, "outer")

    def left(self, other_operand: typing.Union[str, FeatureSet], asof_join):
        """
        Specifies a left join with the given feature set

        :param other_operand:       (str or FeatureSet) The name of the feature set or a FeatureSet object to join with.
        :param asof_join:           (bool) A flag indicating whether to perform an as-of join.

        :return:                    JoinGraph: The updated JoinGraph object with the specified left join.
        """
        return self._join_operands(other_operand, "left", asof_join=asof_join)

    def right(self, other_operand: typing.Union[str, FeatureSet]):
        """
        Specifies a right join with the given feature set

        :param other_operand:       (str or FeatureSet) The name of the feature set or a FeatureSet object to join with.

        :return:                    JoinGraph: The updated JoinGraph object with the specified right join.
        """
        return self._join_operands(other_operand, "right")

    def _join_operands(
        self,
        other_operand: typing.Union[str, FeatureSet],
        join_type: str,
        asof_join: bool = False,
    ):
        if isinstance(other_operand, FeatureSet):
            other_operand = other_operand.metadata.name

        first_key_num = len(self._steps.keys()) if self._steps else 0
        left_last_step_name, left_all_feature_sets = (
            self.last_step_name,
            self.all_feature_sets_names,
        )
        is_first_fs = (
            join_type == JoinGraph.first_join_type or left_all_feature_sets == self.name
        )
        # create_new_step
        new_step = _JoinStep(
            f"step_{first_key_num}",
            left_last_step_name if not is_first_fs else "",
            other_operand,
            left_all_feature_sets if not is_first_fs else [],
            other_operand,
            join_type,
            asof_join,
        )

        if self.steps is not None:
            self.steps.update(new_step)
        else:
            self.steps = [new_step]
        return self

    def _start(self, other_operand: typing.Union[str, FeatureSet]):
        return self._join_operands(other_operand, JoinGraph.first_join_type)

    def _init_all_join_keys(
        self,
        feature_set_objects,
        vector,
        entity_rows_keys: typing.Optional[list[str]] = None,
    ):
        for step in self.steps:
            step.init_join_keys(feature_set_objects, vector, entity_rows_keys)

    @property
    def all_feature_sets_names(self):
        """
         Returns a list of all feature set names included in the join graph.

        :return:                    List[str]: A list of feature set names.
        """
        if self._steps:
            return self._steps[-1].left_feature_set_names + [
                self._steps[-1].right_feature_set_name
            ]
        else:
            return self.name

    @property
    def last_step_name(self):
        """
        Returns the name of the last step in the join graph.

        :return:                    str: The name of the last step.
        """
        if self._steps:
            return self._steps[-1].name
        else:
            return self.name

    @property
    def steps(self):
        """
        Returns the list of join steps as ObjectList, which can be used to iterate over the steps
        or access the properties of each step.
        :return:                    ObjectList: The list of join steps.
        """
        return self._steps

    @steps.setter
    def steps(self, steps):
        """
         Setter for the steps property. It allows updating the join steps.

        :param steps:               (List[_JoinStep]) The list of join steps.
        """
        self._steps = ObjectList.from_list(child_class=_JoinStep, children=steps)


class OnlineVectorService:
    """get_online_feature_service response object"""

    def __init__(
        self,
        vector,
        graph,
        index_columns,
        impute_policy: typing.Optional[dict] = None,
        requested_columns: typing.Optional[list[str]] = None,
    ):
        self.vector = vector
        self.impute_policy = impute_policy or {}

        self._controller = graph.controller
        self._index_columns = index_columns
        self._impute_values = {}
        self._requested_columns = requested_columns

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def initialize(self):
        """internal, init the feature service and prep the imputing logic"""
        if not self.impute_policy:
            return

        impute_policy = copy(self.impute_policy)
        vector = self.vector
        feature_stats = vector.get_stats_table()
        self._impute_values = {}

        feature_keys = list(vector.status.features.keys())
        if vector.status.label_column in feature_keys:
            feature_keys.remove(vector.status.label_column)

        if "*" in impute_policy:
            value = impute_policy["*"]
            del impute_policy["*"]

            for name in feature_keys:
                if name not in impute_policy:
                    if isinstance(value, str) and value.startswith("$"):
                        self._impute_values[name] = feature_stats.loc[name, value[1:]]
                    else:
                        self._impute_values[name] = value

        for name, value in impute_policy.items():
            if name not in feature_keys:
                raise mlrun.errors.MLRunInvalidArgumentError(
                    f"feature {name} in impute_policy but not in feature vector"
                )
            if isinstance(value, str) and value.startswith("$"):
                self._impute_values[name] = feature_stats.loc[name, value[1:]]
            else:
                self._impute_values[name] = value

    @property
    def status(self):
        """vector merger function status (ready, running, error)"""
        return "ready"

    def get(self, entity_rows: list[Union[dict, list]], as_list=False):
        """get feature vector given the provided entity inputs

        take a list of input vectors/rows and return a list of enriched feature vectors
        each input and/or output vector can be a list of values or a dictionary of field names and values,
        to return the vector as a list of values set the `as_list` to True.

        if the input is a list of list (vs a list of dict), the values in the list will correspond to the
        index/entity values, i.e. [["GOOG"], ["MSFT"]] means "GOOG" and "MSFT" are the index/entity fields.

        example::

            # accept list of dict, return list of dict
            svc = fstore.get_online_feature_service(vector)
            resp = svc.get([{"name": "joe"}, {"name": "mike"}])

            # accept list of list, return list of list
            svc = fstore.get_online_feature_service(vector, as_list=True)
            resp = svc.get([["joe"], ["mike"]])

        :param entity_rows:  list of list/dict with input entity data/rows
        :param as_list:      return a list of list (list input is required by many ML frameworks)
        """
        results = []
        futures = []
        if isinstance(entity_rows, dict):
            entity_rows = [entity_rows]

        # validate we have valid input struct
        if (
            not entity_rows
            or not isinstance(entity_rows, list)
            or not isinstance(entity_rows[0], (list, dict))
        ):
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"input data is of type {type(entity_rows)}. must be a list of lists or list of dicts"
            )

        # if list of list, convert to dicts (with the index columns as the dict keys)
        if isinstance(entity_rows[0], list):
            if not self._index_columns or len(entity_rows[0]) != len(
                self._index_columns
            ):
                raise mlrun.errors.MLRunInvalidArgumentError(
                    "input list must be in the same size of the index_keys list"
                )
            index_range = range(len(self._index_columns))
            entity_rows = [
                {self._index_columns[i]: item[i] for i in index_range}
                for item in entity_rows
            ]

        for row in entity_rows:
            futures.append(self._controller.emit(row, return_awaitable_result=True))

        for future in futures:
            result = future.await_result()
            data = result.body
            if data:
                actual_columns = data.keys()
                if all([col in self._index_columns for col in actual_columns]):
                    # didn't get any data from the graph
                    results.append(None)
                    continue
                for column in self._requested_columns:
                    if (
                        column not in actual_columns
                        and column != self.vector.status.label_column
                    ):
                        data[column] = None

                if self._impute_values:
                    for name in data.keys():
                        v = data[name]
                        if v is None or (
                            isinstance(v, float) and (np.isinf(v) or np.isnan(v))
                        ):
                            data[name] = self._impute_values.get(name, v)
                if not self.vector.spec.with_indexes:
                    for name in self.vector.status.index_keys:
                        data.pop(name, None)
                if not any(data.values()):
                    data = None

            if as_list and data:
                data = [
                    data.get(key, None)
                    for key in self._requested_columns
                    if key != self.vector.status.label_column
                ]
            results.append(data)

        return results

    def close(self):
        """terminate the async loop"""
        self._controller.terminate()


class OfflineVectorResponse:
    """get_offline_features response object"""

    def __init__(self, merger):
        self._merger = merger
        self.vector = merger.vector

    @property
    def status(self):
        """vector prep job status (ready, running, error)"""
        return self._merger.get_status()

    def to_dataframe(self, to_pandas=True):
        """return result as dataframe"""
        if self.status != "completed":
            raise mlrun.errors.MLRunTaskNotReadyError(
                "feature vector dataset is not ready"
            )
        return self._merger.get_df(to_pandas=to_pandas)

    def to_parquet(self, target_path, **kw):
        """return results as parquet file"""
        return self._merger.to_parquet(target_path, **kw)

    def to_csv(self, target_path, **kw):
        """return results as csv file"""
        return self._merger.to_csv(target_path, **kw)
