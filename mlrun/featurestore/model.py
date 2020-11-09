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

from typing import Dict, List, Optional
from mlrun.model import ModelObj
from .datatypes import ValueType
from ..model import ObjectList
from ..serving.states import ServingTaskState, ServingRootFlowState


class FeatureClassKind:
    FeatureVector = "FeatureVector"
    FeatureSet = "FeatureSet"
    Entity = "Entity"


class Entity(ModelObj):
    def __init__(
        self,
        name: str = None,
        value_type: ValueType = None,
        description: str = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.description = description
        self.value_type = value_type
        self.labels = labels or {}


class Feature(ModelObj):
    _dict_fields = [
        "name",
        "description",
        "value_type",
        "shape",
        "default",
        "labels",
        "windows",
        "operations",
        "period",
    ]

    def __init__(self, value_type: ValueType = None, description=None, name=None):
        self.name = name or ""
        self.value_type: ValueType = value_type or ""
        self.shape = None
        self.description = description
        self.default = None
        self.labels = {}

        # aggregated features
        self.windows = None
        self.operations = None
        self.period = None


class FeatureSetProducer(ModelObj):
    def __init__(self, kind=None, name=None, uri=None, owner=None):
        self.kind = kind
        self.name = name
        self.owner = owner
        self.uri = uri
        self.sources = {}


class TargetTypes:
    parquet = "parquet"
    nosql = "nosql"
    tsdb = "tsdb"
    stream = "stream"


def is_online_store(target_type):
    return target_type in [TargetTypes.nosql]


def is_offline_store(target_type, with_timestamp=True):
    return target_type in [TargetTypes.parquet, TargetTypes.tsdb] or (
        with_timestamp and target_type == TargetTypes.nosql
    )


def get_offline_store(type_list, requested_type):
    if requested_type:
        if requested_type in type_list and is_offline_store(requested_type):
            return requested_type
        raise ValueError(
            f"target type {requested_type}, not available or is not offline type"
        )
    # todo: sort from best (e.g. parquet) to last
    if TargetTypes.parquet in type_list:
        return TargetTypes.parquet
    for value in type_list:
        if is_offline_store(value):
            return value
    raise ValueError("did not find a valid offline features table")


def get_online_store(type_list):
    if TargetTypes.nosql in type_list:
        return TargetTypes.nosql
    raise ValueError("did not find a valid offline features table")


class DataTarget(ModelObj):
    _dict_fields = ["name", "path", "start_time", "num_rows", "size", "status"]

    def __init__(self, name: TargetTypes = None, path=None):
        self.name: TargetTypes = name
        self.status = ""
        self.updated = None
        self.path = path
        self.max_age = None
        self.start_time = None
        self.num_rows = None
        self.size = None
        self._producer = None
        self.producer = {}

    @property
    def producer(self) -> FeatureSetProducer:
        return self._producer

    @producer.setter
    def producer(self, producer):
        self._producer = self._verify_dict(producer, "producer", FeatureSetProducer)


class FeatureAggregation(ModelObj):
    def __init__(
        self, name=None, column=None, operations=None, windows=None, period=None
    ):
        self.name = name
        self.column = column
        self.operations = operations or []
        self.windows = windows or []
        self.period = period


class FeatureSetMetadata(ModelObj):
    def __init__(
        self,
        name=None,
        tag=None,
        hash=None,
        project=None,
        labels=None,
        annotations=None,
        updated=None,
    ):
        self.name = name
        self.tag = tag
        self.hash = hash
        self.project = project
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.updated = updated


class FeatureSetSpec(ModelObj):
    def __init__(
        self,
        description=None,
        entities=None,
        features=None,
        partition_keys=None,
        timestamp_key=None,
        label_column=None,
        relations=None,
        aggregations=None,
        flow=None,
    ):
        self._features: ObjectList = None
        self._entities: ObjectList = None
        self._aggregations = None
        self._flow: ServingRootFlowState = None

        self.description = description
        self.entities = entities or []
        self.features: List[Feature] = features or []
        self.aggregations: List[FeatureAggregation] = aggregations or []
        self.partition_keys = partition_keys or []
        self.timestamp_key = timestamp_key
        self.relations = relations or {}
        self.flow = flow
        self.label_column = label_column

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
    def aggregations(self) -> List[FeatureAggregation]:
        return self._aggregations

    @aggregations.setter
    def aggregations(self, aggregations: List[FeatureAggregation]):
        self._aggregations = ObjectList.from_list(FeatureAggregation, aggregations)

    @property
    def flow(self) -> ServingRootFlowState:
        return self._flow

    @flow.setter
    def flow(self, flow):
        self._flow = self._verify_dict(flow, "flow", ServingRootFlowState)

    def require_processing(self):
        return len(self._flow.states) > 0 or len(self._aggregations) > 0


class FeatureSetStatus(ModelObj):
    def __init__(self, state=None, targets=None, stats=None, preview=None):
        self.state = state or "created"
        self._targets: ObjectList = None
        self.targets = targets or []
        self.stats = stats or {}
        self.preview = preview or []

    @property
    def targets(self) -> List[DataTarget]:
        return self._targets

    @targets.setter
    def targets(self, targets: List[DataTarget]):
        self._targets = ObjectList.from_list(DataTarget, targets)

    def update_target(self, target: DataTarget):
        self._targets.update(target)


class FeatureVectorSpec(ModelObj):
    _dict_fields = [
        "features",
        "description",
        "entity_source",
        "target_path",
        "flow",
        "label_column",
    ]

    def __init__(
        self,
        client=None,
        features=None,
        description=None,
        entity_source=None,
        target_path=None,
        flow=None,
        label_column=None,
    ):
        self.description = description
        self.features: List[str] = features or []
        self.entity_source = entity_source
        self.target_path = target_path
        self.flow = flow or []
        self.label_column = label_column


class FeatureVectorStatus(ModelObj):
    def __init__(self, state=None, target=None, stats=None, preview=None):
        self.state = state or "created"
        self._target: DataTarget = None
        self.target = target
        self.stats = stats or {}
        self.preview = preview or []

    @property
    def target(self) -> DataTarget:
        return self._spec

    @target.setter
    def target(self, target):
        self._target = self._verify_dict(target, "target", DataTarget)
