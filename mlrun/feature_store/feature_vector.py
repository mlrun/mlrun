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
import collections
import logging
from copy import copy
from enum import Enum
from typing import List, Union

import numpy as np
import pandas as pd

import mlrun

from ..config import config as mlconf
from ..datastore import get_store_uri
from ..datastore.targets import get_offline_target
from ..feature_store.common import (
    get_feature_set_by_uri,
    parse_feature_string,
    parse_project_name_from_feature_string,
)
from ..features import Feature
from ..model import DataSource, DataTarget, ModelObj, ObjectList, VersionedObjMetadata
from ..runtimes.function_reference import FunctionReference
from ..serving.states import RootFlowStep
from ..utils import StorePrefix


class FeatureVectorSpec(ModelObj):
    def __init__(
        self,
        features=None,
        description=None,
        entity_source=None,
        entity_fields=None,
        timestamp_field=None,
        graph=None,
        label_feature=None,
        with_indexes=None,
        function=None,
        analysis=None,
    ):
        self._graph: RootFlowStep = None
        self._entity_fields: ObjectList = None
        self._entity_source: DataSource = None
        self._function: FunctionReference = None

        self.description = description
        self.features: List[str] = features or []
        self.entity_source = entity_source
        self.entity_fields = entity_fields or []
        self.graph = graph
        self.timestamp_field = timestamp_field
        self.label_feature = label_feature
        self.with_indexes = with_indexes
        self.function = function
        self.analysis = analysis or {}

    @property
    def entity_source(self) -> DataSource:
        """data source used as entity source (events/keys need to be enriched)"""
        return self._entity_source

    @entity_source.setter
    def entity_source(self, source: DataSource):
        self._entity_source = self._verify_dict(source, "entity_source", DataSource)

    @property
    def entity_fields(self) -> List[Feature]:
        """the schema/metadata for the entity source fields"""
        return self._entity_fields

    @entity_fields.setter
    def entity_fields(self, entity_fields: List[Feature]):
        self._entity_fields = ObjectList.from_list(Feature, entity_fields)

    @property
    def graph(self) -> RootFlowStep:
        """feature vector transformation graph/DAG"""
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = self._verify_dict(graph, "graph", RootFlowStep)
        self._graph.engine = "async"

    @property
    def function(self) -> FunctionReference:
        """reference to template graph processing function"""
        return self._function

    @function.setter
    def function(self, function):
        self._function = self._verify_dict(function, "function", FunctionReference)


class FeatureVectorStatus(ModelObj):
    def __init__(
        self,
        state=None,
        targets=None,
        features=None,
        label_column=None,
        stats=None,
        preview=None,
        run_uri=None,
        index_keys=None,
    ):
        self._targets: ObjectList = None
        self._features: ObjectList = None

        self.state = state or "created"
        self.label_column = label_column
        self.targets = targets
        self.stats = stats or {}
        self.index_keys = index_keys
        self.preview = preview or []
        self.features: List[Feature] = features or []
        self.run_uri = run_uri

    @property
    def targets(self) -> List[DataTarget]:
        """list of material storage targets + their status/path"""
        return self._targets

    @targets.setter
    def targets(self, targets: List[DataTarget]):
        self._targets = ObjectList.from_list(DataTarget, targets)

    def update_target(self, target: DataTarget):
        self._targets.update(target)

    @property
    def features(self) -> List[Feature]:
        """list of features (result of joining features from the source feature sets)"""
        return self._features

    @features.setter
    def features(self, features: List[Feature]):
        self._features = ObjectList.from_list(Feature, features)


class FeatureVector(ModelObj):
    """Feature vector, specify selected features, their metadata and material views"""

    kind = mlrun.api.schemas.ObjectKind.feature_vector.value
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(
        self,
        name=None,
        features=None,
        label_feature=None,
        description=None,
        with_indexes=None,
    ):
        """Feature vector, specify selected features, their metadata and material views

        example::

            import mlrun.feature_store as fstore
            features = ["quotes.bid", "quotes.asks_sum_5h as asks_5h", "stocks.*"]
            vector = fstore.FeatureVector("my-vec", features)

            # get the vector as a dataframe
            df = fstore.get_offline_features(vector).to_dataframe()

            # return an online/real-time feature service
            svc = fstore.get_online_feature_service(vector, impute_policy={"*": "$mean"})
            resp = svc.get([{"stock": "GOOG"}])

        :param name:           List of names of targets to delete (default: delete all ingested targets)
        :param features:       list of feature to collect to this vector.
                                Format [<project>/]<feature_set>.<feature_name or `*`> [as <alias>]
        :param label_feature:  feature name to be used as label data
        :param description:    text description of the vector
        :param with_indexes:   whether to keep the entity and timestamp columns in the response
        """
        self._spec: FeatureVectorSpec = None
        self._metadata = None
        self._status = None

        self.spec = FeatureVectorSpec(
            description=description,
            features=features,
            label_feature=label_feature,
            with_indexes=with_indexes,
        )
        self.metadata = VersionedObjMetadata(name=name)
        self.status = None

        self._entity_df = None
        self._feature_set_fields = {}
        self.feature_set_objects = {}

    @property
    def spec(self) -> FeatureVectorSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", FeatureVectorSpec)

    @property
    def metadata(self) -> VersionedObjMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", VersionedObjMetadata)

    @property
    def status(self) -> FeatureVectorStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", FeatureVectorStatus)

    @property
    def uri(self):
        """fully qualified feature vector uri"""
        uri = (
            f"{self._metadata.project or mlconf.default_project}/{self._metadata.name}"
        )
        uri = get_store_uri(StorePrefix.FeatureVector, uri)
        if self._metadata.tag:
            uri += ":" + self._metadata.tag
        return uri

    def link_analysis(self, name, uri):
        """add a linked file/artifact (chart, data, ..)"""
        self._spec.analysis[name] = uri

    def get_stats_table(self):
        """get feature statistics table (as dataframe)"""
        if self.status.stats:
            feature_aliases = self.get_feature_aliases()
            for old_name, new_name in feature_aliases.items():
                if old_name in self.status.stats:
                    self.status.stats[new_name] = self.status.stats[old_name]
                    del self.status.stats[old_name]
            return pd.DataFrame.from_dict(self.status.stats, orient="index")

    def get_feature_aliases(self):
        feature_aliases = {}
        for feature in self.spec.features:
            column_names = feature.split(" as ")
            # split 'feature_set.old_name as new_name'
            if len(column_names) == 2:
                old_name_with_feature_set, new_name = column_names
                # split 'feature_set.old_name'
                feature_set, old_name = column_names[0].split(".")
                if new_name != old_name:
                    feature_aliases[old_name] = new_name
        return feature_aliases

    def get_target_path(self, name=None):
        target = get_offline_target(self, name=name)
        if target:
            return target.path

    def to_dataframe(self, df_module=None, target_name=None):
        """return feature vector (offline) data as dataframe"""
        driver = get_offline_target(self, name=target_name)
        if not driver:
            raise mlrun.errors.MLRunNotFoundError(
                "there are no offline targets for this feature vector"
            )
        return driver.as_df(df_module=df_module)

    def save(self, tag="", versioned=False):
        """save to mlrun db"""
        db = mlrun.get_run_db()
        self.metadata.project = self.metadata.project or mlconf.default_project
        tag = tag or self.metadata.tag
        as_dict = self.to_dict()
        db.store_feature_vector(as_dict, tag=tag, versioned=versioned)

    def reload(self, update_spec=True):
        """reload/sync the feature set status and spec from the DB"""
        from_db = mlrun.get_run_db().get_feature_vector(
            self.metadata.name, self.metadata.project, self.metadata.tag
        )
        self.status = from_db.status
        if update_spec:
            self.spec = from_db.spec

    def parse_features(self, offline=True, update_stats=False):
        """parse and validate feature list (from vector) and add metadata from feature sets

        :returns
            feature_set_objects: cache of used feature set objects
            feature_set_fields:  list of field (name, alias) per featureset
        """
        processed_features = {}  # dict of name to (featureset, feature object)
        feature_set_objects = {}
        index_keys = []
        feature_set_fields = collections.defaultdict(list)
        features = copy(self.spec.features)
        label_column_name = None
        label_column_fset = None
        if offline and self.spec.label_feature:
            features.append(self.spec.label_feature)
            feature_set, name, _ = parse_feature_string(self.spec.label_feature)
            self.status.label_column = name
            label_column_name = name
            label_column_fset = feature_set

        def add_feature(name, alias, feature_set_object, feature_set_full_name):
            if alias in processed_features.keys():
                logging.log(
                    logging.WARN,
                    f"feature name/alias {alias} already specified,"
                    " you need to use another alias (feature-set.name [as alias])"
                    f" by default it changed to be {alias}_{feature_set_full_name}",
                )
                alias = f"{alias}_{feature_set_full_name}"

            feature = feature_set_object[name]
            processed_features[alias or name] = (feature_set_object, feature)
            feature_set_fields[feature_set_full_name].append((name, alias))

        for feature in features:
            project_name, feature = parse_project_name_from_feature_string(feature)
            feature_set, feature_name, alias = parse_feature_string(feature)
            if feature_set not in feature_set_objects.keys():
                feature_set_objects[feature_set] = get_feature_set_by_uri(
                    feature_set,
                    project_name if project_name is not None else self.metadata.project,
                )
            feature_set_object = feature_set_objects[feature_set]

            feature_fields = feature_set_object.spec.features.keys()
            if feature_name == "*":
                for field in feature_fields:
                    if field != feature_set_object.spec.timestamp_key and not (
                        feature_set == label_column_fset and field == label_column_name
                    ):
                        if alias:
                            add_feature(
                                field,
                                alias + "_" + field,
                                feature_set_object,
                                feature_set,
                            )
                        else:
                            add_feature(field, field, feature_set_object, feature_set)
            else:
                if feature_name not in feature_fields:
                    raise mlrun.errors.MLRunInvalidArgumentError(
                        f"feature {feature} not found in feature set {feature_set}"
                    )
                add_feature(feature_name, alias, feature_set_object, feature_set)

        for feature_set_name, fields in feature_set_fields.items():
            feature_set = feature_set_objects[feature_set_name]
            for key in feature_set.spec.entities.keys():
                if key not in index_keys:
                    index_keys.append(key)
            for name, _ in fields:
                if name in feature_set.status.stats and update_stats:
                    self.status.stats[name] = feature_set.status.stats[name]
                if name in feature_set.spec.features.keys():
                    feature = feature_set.spec.features[name].copy()
                    feature.origin = f"{feature_set.fullname}.{name}"
                    self.status.features[name] = feature

        self.status.index_keys = index_keys
        return feature_set_objects, feature_set_fields


class OnlineVectorService:
    """get_online_feature_service response object"""

    def __init__(self, vector, graph, index_columns, impute_policy: dict = None):
        self.vector = vector
        self.impute_policy = impute_policy or {}

        self._controller = graph.controller
        self._index_columns = index_columns
        self._impute_values = {}

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

    def get(self, entity_rows: List[Union[dict, list]], as_list=False):
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

        requested_columns = list(self.vector.status.features.keys())
        aliases = self.vector.get_feature_aliases()
        for i, column in enumerate(requested_columns):
            requested_columns[i] = aliases.get(column, column)

        for future in futures:
            result = future.await_result()
            data = result.body
            for key in self._index_columns:
                if data and key in data:
                    del data[key]
            if not data:
                data = None
            else:
                actual_columns = data.keys()
                for column in requested_columns:
                    if (
                        column not in actual_columns
                        and column != self.vector.status.label_column
                    ):
                        data[column] = None

            if self._impute_values and data:
                for name in data.keys():
                    v = data[name]
                    if v is None or (type(v) == float and (np.isinf(v) or np.isnan(v))):
                        data[name] = self._impute_values.get(name, v)

            if as_list and data:
                data = [
                    data.get(key, None)
                    for key in requested_columns
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
            raise mlrun.errors.MLRunTaskNotReady("feature vector dataset is not ready")
        return self._merger.get_df(to_pandas=to_pandas)

    def to_parquet(self, target_path, **kw):
        """return results as parquet file"""
        return self._merger.to_parquet(target_path, **kw)

    def to_csv(self, target_path, **kw):
        """return results as csv file"""
        return self._merger.to_csv(target_path, **kw)


class FixedWindowType(Enum):
    CurrentOpenWindow = 1
    LastClosedWindow = 2

    def to_qbk_fixed_window_type(self):
        try:
            from storey import FixedWindowType as QueryByKeyFixedWindowType
        except ImportError as exc:
            raise ImportError("storey not installed, use pip install storey") from exc
        if self == FixedWindowType.LastClosedWindow:
            return QueryByKeyFixedWindowType.LastClosedWindow
        elif self == FixedWindowType.CurrentOpenWindow:
            return QueryByKeyFixedWindowType.CurrentOpenWindow
        else:
            raise NotImplementedError(
                f"Provided fixed window type is not supported. fixed_window_type={self}"
            )
