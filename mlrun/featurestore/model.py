from typing import Dict, List, Optional
from mlrun.model import ModelObj
from .datatypes import ValueType
from ..model import ObjectList


class FeatureClassKind:
    FeatureVector = 'featurevector'
    FeatureSet = 'featureset'
    Entity = 'entity'


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
    def __init__(self, value_type: ValueType = None, description=None, name=None):
        self.name = name or ''
        self.value_type: ValueType = value_type or ''
        self.shape = None
        self.description = description
        self.default = None
        self.labels = {}


class FeatureSetProducer(ModelObj):
    def __init__(self, kind=None, name=None, uri=None, owner=None):
        self.kind = kind
        self.name = name
        self.owner = owner
        self.uri = uri or "/"
        self.sources = {}


class TargetTypes:
    parquet = 'parquet'
    kv = 'kv'
    tsdb = 'tsdb'
    stream = 'stream'


def is_online_store(target_type):
    return target_type in [TargetTypes.kv]


def is_offline_store(target_type):
    return target_type in [TargetTypes.parquet, TargetTypes.tsdb]


def get_offline_store(type_list, requested_type):
    if requested_type:
        if requested_type in type_list and is_offline_store(requested_type):
            return requested_type
        raise ValueError(f'target type {requested_type}, not available or is not offline type')
    for value in type_list:
        if is_offline_store(value):
            return value
    raise ValueError('did not find a valid offline features table')


def get_online_store(type_list):
    if TargetTypes.kv in type_list:
        return TargetTypes.kv
    raise ValueError('did not find a valid offline features table')


class DataTarget(ModelObj):
    def __init__(self, name: TargetTypes = None, path=None):
        self.name: TargetTypes = name
        self.status = ''
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
        self._producer = self._verify_dict(producer, 'producer', FeatureSetProducer)


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
    def __init__(self, description=None, entities=None, features=None,
                 primary_keys=None, partition_keys=None,
                 timestamp_key=None, label_column=None, relations=None):
        self._features: ObjectList = None
        self._entities: ObjectList = None

        self.description = description
        self.entities = entities or []
        self.features: List[Feature] = features or []
        self.primary_keys = primary_keys or []
        self.partition_keys = partition_keys or []
        self.timestamp_key = timestamp_key
        self.relations = relations or {}
        self.label_column = label_column

    @property
    def entities(self) -> List[Entity]:
        return self._entities.to_list()

    @entities.setter
    def entities(self, entities: List[Entity]):
        self._entities = ObjectList.from_list(Entity, entities)

    def get_entities_map(self):
        return self._entities

    @property
    def features(self) -> List[Feature]:
        return self._features.to_list()

    @features.setter
    def features(self, features: List[Feature]):
        self._features = ObjectList.from_list(Feature, features)

    def get_features_map(self):
        return self._features


class FeatureSetStatus(ModelObj):
    def __init__(self, state=None, targets=None, stats=None):
        self.state = state or 'created'
        self._targets: ObjectList = None
        self.targets = targets or []
        self.stats = stats or {}
        self.preview = []

    @property
    def targets(self) -> List[DataTarget]:
        return self._targets.to_list()

    @targets.setter
    def targets(self, targets: List[DataTarget]):
        self._targets = ObjectList.from_list(DataTarget, targets)

    def update_target(self, target: DataTarget):
        self._targets.update(target)

    def get_targets_map(self):
        return self._targets
