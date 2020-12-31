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
from typing import Dict, Optional

from mlrun.model import ModelObj
from .datatypes import ValueType
from .validators import validator_types


class FeatureStoreError(Exception):
    """error in feature store data or configuration"""

    pass


class TargetTypes:
    csv = "csv"
    parquet = "parquet"
    nosql = "nosql"
    tsdb = "tsdb"
    stream = "stream"
    dataframe = "dataframe"


class ResourceKinds:
    FeatureSet = "FeatureSet"
    FeatureVector = "FeatureVector"


default_config = {
    "data_prefixes": {
        "default": "./store/{project}/{kind}",
        "parquet": "v3io:///projects/{project}/fs/{kind}",
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
    _dict_fields = ["name", "kind", "path", "after_state", "attributes"]

    def __init__(
        self, kind: TargetTypes = None, name: str = "", path=None, after_state=None
    ):
        self.name = name
        self.kind: TargetTypes = kind
        self.path = path
        self.after_state = after_state
        self.attributes = None
        self.driver = None
        self._table = None

    def set_table(self, table):
        self._table = table

    @property
    def table(self):
        return self._table


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
    _dict_fields = [
        "name",
        "kind",
        "path",
        "attributes",
        "online",
        "workers",
        "max_age",
    ]

    def __init__(
        self, name: str = "", kind: TargetTypes = None, path=None, online=None
    ):
        self.name = name
        self.online = online
        self.kind: SourceTypes = kind
        self.path = path
        self.max_age = None
        self.attributes = None
        self.workers = None


class FeatureAggregation(ModelObj):
    def __init__(
        self, name=None, column=None, operations=None, windows=None, period=None
    ):
        self.name = name
        self.column = column
        self.operations = operations or []
        self.windows = windows or []
        self.period = period


class CommonMetadata(ModelObj):
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
