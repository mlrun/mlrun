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
from copy import copy
from typing import List

import mlrun
import pandas as pd

from .base import (
    FeatureAggregation,
    Feature,
    Entity,
    DataTarget,
    DataSource,
    ResourceKinds,
    FeatureStoreError,
    DataTargetSpec,
    store_config,
    CommonMetadata,
)
from ..targets import get_offline_target
from mlrun.model import ModelObj, ObjectList
from mlrun.runtimes.function_reference import FunctionReference
from mlrun.serving.states import BaseState, RootFlowState
from mlrun.config import config as mlconf
from mlrun.utils import get_store_uri, StorePrefix

aggregates_step = "Aggregates"


class FeatureSetSpec(ModelObj):
    def __init__(
        self,
        owner=None,
        description=None,
        entities=None,
        features=None,
        partition_keys=None,
        timestamp_key=None,
        label_column=None,
        relations=None,
        source=None,
        targets=None,
        graph=None,
        function=None,
    ):
        self._features: ObjectList = None
        self._entities: ObjectList = None
        self._targets: ObjectList = None
        self._graph: RootFlowState = None
        self._source = None
        self._function: FunctionReference = None

        self.owner = owner
        self.description = description
        self.entities: List[Entity] = entities or []
        self.features: List[Feature] = features or []
        self.partition_keys = partition_keys or []
        self.timestamp_key = timestamp_key
        self.relations = relations or {}
        self.source = source
        self.targets = targets or []
        self.graph = graph
        self.label_column = label_column
        self.function = function

    @property
    def entities(self) -> List[Entity]:
        return self._entities

    @entities.setter
    def entities(self, entities: List[Entity]):
        self._entities = ObjectList.from_list(Entity, entities)

    @property
    def features(self) -> List[Feature]:
        return self._features

    @features.setter
    def features(self, features: List[Feature]):
        self._features = ObjectList.from_list(Feature, features)

    @property
    def targets(self) -> List[DataTargetSpec]:
        return self._targets

    @targets.setter
    def targets(self, targets: List[DataTargetSpec]):
        self._targets = ObjectList.from_list(DataTargetSpec, targets)

    @property
    def graph(self) -> RootFlowState:
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = self._verify_dict(graph, "graph", RootFlowState)
        self._graph.engine = "async"

    @property
    def function(self) -> FunctionReference:
        return self._spec

    @function.setter
    def function(self, function):
        self._function = self._verify_dict(function, "function", FunctionReference)

    @property
    def source(self) -> DataSource:
        return self._source

    @source.setter
    def source(self, source: DataSource):
        self._source = self._verify_dict(source, "source", DataSource)

    def require_processing(self):
        return len(self._graph.states) > 0


class FeatureSetStatus(ModelObj):
    def __init__(
        self,
        state=None,
        targets=None,
        stats=None,
        preview=None,
        runtime=None,
        run_uri=None,
    ):
        self.state = state or "created"
        self._targets: ObjectList = None
        self.targets = targets or []
        self.stats = stats or {}
        self.preview = preview or []
        self.runtime = runtime
        self.run_uri = run_uri

    @property
    def targets(self) -> List[DataTarget]:
        return self._targets

    @targets.setter
    def targets(self, targets: List[DataTarget]):
        self._targets = ObjectList.from_list(DataTarget, targets)

    def update_target(self, target: DataTarget):
        self._targets.update(target)


class FeatureSet(ModelObj):
    """Feature Set"""

    kind = ResourceKinds.FeatureSet
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(self, name=None, description=None, entities=None, timestamp_key=None):
        self._spec: FeatureSetSpec = None
        self._metadata = None
        self._status = None
        self._api_client = None

        self.spec = FeatureSetSpec(
            description=description, entities=entities, timestamp_key=timestamp_key
        )
        self.metadata = CommonMetadata(name=name)
        self.status = None
        self._last_state = ""

    @property
    def spec(self) -> FeatureSetSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", FeatureSetSpec)

    @property
    def metadata(self) -> CommonMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(metadata, "metadata", CommonMetadata)

    @property
    def status(self) -> FeatureSetStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", FeatureSetStatus)

    def uri(self):
        uri = f'{self._metadata.project or ""}/{self._metadata.name}'
        uri = get_store_uri(StorePrefix.FeatureSet, uri)
        if self._metadata.tag:
            uri += ":" + self._metadata.tag
        return uri

    def set_targets(self, targets=None):
        if targets is not None and not isinstance(targets, list):
            raise ValueError(
                "targets can only be None or a list of kinds/DataTargetSpec"
            )
        targets = targets or copy(store_config.default_targets)
        for target in targets:
            if not isinstance(target, DataTargetSpec):
                target = DataTargetSpec(target, name=str(target))
            self.spec.targets.update(target)

    def add_entity(self, entity, name=None):
        self._spec.entities.update(entity, name)

    def add_feature(self, feature, name=None):
        self._spec.features.update(feature, name)

    @property
    def graph(self):
        return self.spec.graph

    def add_aggregation(
        self,
        name,
        column,
        operations,
        windows,
        period=None,
        state_name=None,
        after=None,
        before=None,
    ):
        aggregation = FeatureAggregation(
            name, column, operations, windows, period
        ).to_dict()

        def upsert_feature(name):
            if name in self.spec.features:
                self.spec.features[name].aggregate = True
            else:
                self.spec.features[name] = Feature(name=column, aggregate=True)

        state_name = state_name or aggregates_step
        graph = self.spec.graph
        if state_name in graph.states:
            state = graph.states[state_name]
            aggregations = state.class_args.get("aggregates", [])
            aggregations.append(aggregation)
            state.class_args["aggregates"] = aggregations
        else:
            graph.add_step(
                name=state_name,
                after=after or "$prev",
                before=before,
                class_name="storey.AggregateByKey",
                aggregates=[aggregation],
                table=".",
            )

        for operation in operations:
            for window in windows:
                upsert_feature(f"{name}_{operation}_{window}")

    def get_stats_table(self):
        if self.status.stats:
            return pd.DataFrame.from_dict(self.status.stats, orient="index")

    def __getitem__(self, name):
        return self._spec.features[name]

    def __setitem__(self, key, item):
        self._spec.features.update(item, key)

    def plot(self, filename=None, format=None, with_targets=False, **kw):
        graph = self.spec.graph
        targets = None
        if with_targets:
            targets = [
                BaseState(target.kind, after=target.after_state, shape="cylinder")
                for target in self.spec.targets
            ]
        return graph.plot(filename, format, targets=targets, **kw)

    def to_dataframe(self, columns=None, df_module=None, target_name=None):
        if columns:
            if self.spec.timestamp_key:
                columns = [self.spec.timestamp_key] + columns
            columns = list(self.spec.entities.keys()) + columns
        target, driver = get_offline_target(self, name=target_name)
        if not target:
            raise FeatureStoreError("there are no offline targets for this feature set")
        return driver.as_df(columns=columns, df_module=df_module)

    def save(self, tag="", versioned=False):
        db = mlrun.get_run_db()
        self.metadata.project = self.metadata.project or mlconf.default_project
        tag = tag or self.metadata.tag
        as_dict = self.to_dict()
        as_dict["spec"]["features"] = as_dict["spec"].get(
            "features", []
        )  # bypass DB bug
        db.store_feature_set(as_dict, tag=tag, versioned=versioned)
