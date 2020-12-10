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
import copy
from typing import Dict, List, Optional
from mlrun.model import ModelObj
from .datatypes import ValueType
from .validators import validator_types
from ..model import ObjectList
from ..serving.states import ServingRootFlowState


class FeatureClassKind:
    FeatureVector = "FeatureVector"
    FeatureSet = "FeatureSet"
    Entity = "Entity"


class TargetTypes:
    csv = "csv"
    parquet = "parquet"
    nosql = "nosql"
    tsdb = "tsdb"
    stream = "stream"
    dataframe = "dataframe"


default_config = {
    "data_prefixes": {
        "default": "./store/{project}/{kind}",
        "parquet": "./store/{project}/{kind}",
        "nosql": "v3io:///projects/{project}/fs/{kind}",
    },
    "default_targets": [TargetTypes.parquet, TargetTypes.nosql],
}


class FeatureStoreConfig:
    def __init__(self, config=None):
        object.__setattr__(self, "_config", config or {})

    def __getattr__(self, attr):
        val = self._config.get(attr, None)
        if val is None:
            raise AttributeError(attr)
        return val

    def __setattr__(self, attr, value):
        self._config[attr] = value


store_config = FeatureStoreConfig(copy.deepcopy(default_config))


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
        if name and not value_type:
            self.value_type = ValueType.STRING
        self.labels = labels or {}


class Feature(ModelObj):
    _dict_fields = [
        "name",
        "description",
        "value_type",
        "shape",
        "default",
        "labels",
        "aggregate",
        "validator",
    ]

    def __init__(
        self,
        value_type: ValueType = None,
        description=None,
        aggregate=None,
        name=None,
        validator=None,
    ):
        self.name = name or ""
        self.value_type: ValueType = value_type or ""
        self.shape = None
        self.description = description
        self.default = None
        self.labels = {}
        self.aggregate = aggregate
        self._validator = validator

    @property
    def validator(self):
        return self._validator

    @validator.setter
    def validator(self, validator):
        if isinstance(validator, dict):
            kind = validator.get("kind")
            validator = validator_types[kind].from_dict(validator)
        self._validator = validator


class FeatureSetProducer(ModelObj):
    def __init__(self, kind=None, name=None, uri=None, owner=None):
        self.kind = kind
        self.name = name
        self.owner = owner
        self.uri = uri
        self.sources = {}


class SourceTypes:
    offline = "offline"
    realtime = "realtime"


class DataTargetSpec(ModelObj):
    _dict_fields = ["name", "kind", "path", "after_state", "options"]

    def __init__(
        self, kind: TargetTypes = None, name: str = "", path=None, after_state=None
    ):
        self.name = name
        self.kind: TargetTypes = kind
        self.path = path
        self.after_state = after_state
        self.options = None
        self.driver = None


class DataTarget(DataTargetSpec):
    _dict_fields = ["name", "kind", "path", "start_time", "online", "status"]

    def __init__(
        self, kind: TargetTypes = None, name: str = "", path=None, online=None
    ):
        super().__init__(name, kind, path)
        self.status = ""
        self.updated = None
        self.online = online
        self.max_age = None
        self.start_time = None
        self._producer = None
        self.producer = {}

    @property
    def producer(self) -> FeatureSetProducer:
        return self._producer

    @producer.setter
    def producer(self, producer):
        self._producer = self._verify_dict(producer, "producer", FeatureSetProducer)


class DataSource(ModelObj):
    def __init__(self, name: str = "", kind: TargetTypes = None, path=None):
        self.name = name
        self.kind: SourceTypes = kind
        self.path = path
        self.max_age = None


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
        owner=None,
        description=None,
        entities=None,
        features=None,
        partition_keys=None,
        timestamp_key=None,
        label_column=None,
        relations=None,
        sources=None,
        targets=None,
        graph=None,
        final_graph_state=None,
    ):
        self._features: ObjectList = None
        self._entities: ObjectList = None
        self._targets: ObjectList = None
        self._aggregations = None
        self._graph: ServingRootFlowState = None
        self._sources = None

        self.owner = owner
        self.description = description
        self.entities: List[Entity] = entities or []
        self.features: List[Feature] = features or []
        self.partition_keys = partition_keys or []
        self.timestamp_key = timestamp_key
        self.relations = relations or {}
        self.sources = sources or []
        self.targets = targets or []
        self.graph = graph
        self.label_column = label_column
        self.final_graph_state = final_graph_state

    def get_final_state(self):
        return self.final_graph_state or "ValidatorStep"

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
    def graph(self) -> ServingRootFlowState:
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = self._verify_dict(graph, "graph", ServingRootFlowState)
        self._graph.engine = "async"

    @property
    def sources(self) -> List[DataSource]:
        return self._sources

    @sources.setter
    def sources(self, sources: List[DataSource]):
        self._sources = ObjectList.from_list(DataSource, sources)

    def require_processing(self):
        return len(self._graph.states) > 0 or len(self._aggregations) > 0


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
    def __init__(
        self,
        features=None,
        description=None,
        entity_source=None,
        entity_fields=None,
        timestamp_field=None,
        target_path=None,
        graph=None,
        label_column=None,
    ):
        self._graph: ServingRootFlowState = None
        self._entity_fields: ObjectList = None

        self.description = description
        self.features: List[str] = features or []
        self.entity_source = entity_source
        self.entity_fields = entity_fields or []
        self.target_path = target_path
        self.graph = graph
        self.timestamp_field = timestamp_field
        self.label_column = label_column

    @property
    def entity_fields(self) -> List[Feature]:
        return self._entity_fields

    @entity_fields.setter
    def entity_fields(self, entity_fields: List[Feature]):
        self._entity_fields = ObjectList.from_list(Feature, entity_fields)

    @property
    def graph(self) -> ServingRootFlowState:
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = self._verify_dict(graph, "graph", ServingRootFlowState)
        self._graph.engine = "async"


class FeatureVectorStatus(ModelObj):
    def __init__(
        self, state=None, target=None, features=None, stats=None, preview=None
    ):
        self._target: DataTarget = None
        self._features: ObjectList = None

        self.state = state or "created"
        self.target = target
        self.stats = stats or {}
        self.preview = preview or []
        self.features: List[Feature] = features or []

    @property
    def target(self) -> DataTarget:
        return self._spec

    @target.setter
    def target(self, target):
        self._target = self._verify_dict(target, "target", DataTarget)

    @property
    def features(self) -> List[Feature]:
        return self._features

    @features.setter
    def features(self, features: List[Feature]):
        self._features = ObjectList.from_list(Feature, features)
