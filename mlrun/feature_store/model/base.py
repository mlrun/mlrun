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

from typing import Dict, Optional
from mlrun.model import ModelObj
from .data_types import ValueType
from .validators import validator_kinds


class TargetTypes:
    csv = "csv"
    parquet = "parquet"
    nosql = "nosql"
    tsdb = "tsdb"
    stream = "stream"
    dataframe = "dataframe"


class Entity(ModelObj):
    """data entity (index)"""

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
    """data feature"""

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
        default=None,
        labels: Dict[str, str] = None,
    ):
        self.name = name or ""
        self.value_type: ValueType = value_type or ""
        self.shape = None
        self.description = description
        self.default = default
        self.labels = labels or {}
        self.aggregate = aggregate
        self._validator = validator

    @property
    def validator(self):
        return self._validator

    @validator.setter
    def validator(self, validator):
        if isinstance(validator, dict):
            kind = validator.get("kind")
            validator = validator_kinds[kind].from_dict(validator)
        self._validator = validator


class FeatureSetProducer(ModelObj):
    """information about the task/job which produced the feature set data"""

    def __init__(self, kind=None, name=None, uri=None, owner=None, sources=None):
        self.kind = kind
        self.name = name
        self.owner = owner
        self.uri = uri
        self.sources = sources or {}


class SourceTypes:
    csv = "csv"
    dataframe = "dataframe"


class DataTargetBase(ModelObj):
    """data target spec, specify a destination for the feature set data"""

    _dict_fields = ["name", "kind", "path", "after_state", "attributes"]

    def __init__(
        self,
        kind: TargetTypes = None,
        name: str = "",
        path=None,
        attributes: Dict[str, str] = None,
        after_state=None,
    ):
        self.name = name
        self.kind: TargetTypes = kind
        self.path = path
        self.after_state = after_state
        self.attributes = attributes or {}


class DataTarget(DataTargetBase):
    """data target with extra status information (used in the feature-set/vector status)"""

    _dict_fields = ["name", "kind", "path", "start_time", "online", "status"]

    def __init__(
        self, kind: TargetTypes = None, name: str = "", path=None, online=None
    ):
        super().__init__(kind, name, path)
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
    """online or offline data source spec"""

    _dict_fields = [
        "kind",
        "name",
        "path",
        "attributes",
        "key_column",
        "time_column",
        "schedule",
        "online",
        "workers",
        "max_age",
    ]
    kind = None

    def __init__(
        self,
        name: str = None,
        path: str = None,
        attributes: Dict[str, str] = None,
        key_column: str = None,
        time_column: str = None,
        schedule: str = None,
    ):
        self.name = name
        self.path = path
        self.attributes = attributes
        self.schedule = schedule
        self.key_column = key_column
        self.time_column = time_column

        self.online = None
        self.max_age = None
        self.workers = None


class FeatureAggregation(ModelObj):
    """feature aggregation requirements"""

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
        name: str = None,
        tag: str = None,
        uid: str = None,
        project: str = None,
        labels: Dict[str, str] = None,
        annotations: Dict[str, str] = None,
        updated=None,
    ):
        self.name = name
        self.tag = tag
        self.uid = uid
        self.project = project
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.updated = updated
