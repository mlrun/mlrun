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

from dataclasses import dataclass, field
from typing import Any

import mlrun.model
from mlrun.common.model_monitoring.helpers import FeatureStats
from mlrun.common.schemas.model_monitoring.constants import (
    EndpointType,
    EventKeyMetrics,
    EventLiveStats,
    ModelMonitoringMode,
)


@dataclass
class ModelEndpointSpec(mlrun.model.ModelObj):
    function_uri: str = ""  # <project_name>/<function_name>:<tag>
    model: str = ""  # <model_name>:<version>
    model_class: str = ""
    model_uri: str = ""
    feature_names: list[str] = field(default_factory=list)
    label_names: list[str] = field(default_factory=list)
    stream_path: str = ""
    algorithm: str = ""
    monitor_configuration: dict = field(default_factory=dict)
    active: bool = True
    monitoring_mode: ModelMonitoringMode = ModelMonitoringMode.disabled


@dataclass
class ModelEndpointStatus(mlrun.model.ModelObj):
    feature_stats: FeatureStats = field(default_factory=dict)
    current_stats: FeatureStats = field(default_factory=dict)
    first_request: str = ""
    last_request: str = ""
    error_count: int = 0
    drift_status: str = ""
    drift_measures: dict = field(default_factory=dict)
    metrics: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            EventKeyMetrics.GENERIC: {
                EventLiveStats.LATENCY_AVG_1H: 0,
                EventLiveStats.PREDICTIONS_PER_SECOND: 0,
            }
        }
    )
    features: list[dict[str, Any]] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    children_uids: list[str] = field(default_factory=list)
    endpoint_type: EndpointType = EndpointType.NODE_EP
    monitoring_feature_set_uri: str = ""
    state: str = ""


class ModelEndpoint(mlrun.model.ModelObj):
    kind = "model-endpoint"
    _dict_fields = ["kind", "metadata", "spec", "status"]

    def __init__(self):
        self._status: ModelEndpointStatus = ModelEndpointStatus()
        self._spec: ModelEndpointSpec = ModelEndpointSpec()
        self._metadata: mlrun.model.VersionedObjMetadata = (
            mlrun.model.VersionedObjMetadata()
        )

    @property
    def status(self) -> ModelEndpointStatus:
        return self._status

    @status.setter
    def status(self, status):
        self._status = self._verify_dict(status, "status", ModelEndpointStatus)

    @property
    def spec(self) -> ModelEndpointSpec:
        return self._spec

    @spec.setter
    def spec(self, spec):
        self._spec = self._verify_dict(spec, "spec", ModelEndpointSpec)

    @property
    def metadata(self) -> mlrun.model.VersionedObjMetadata:
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        self._metadata = self._verify_dict(
            metadata, "metadata", mlrun.model.VersionedObjMetadata
        )

    @classmethod
    def from_flat_dict(cls, struct=None, fields=None, deprecated_fields: dict = None):
        new_obj = cls()
        new_obj._metadata = mlrun.model.VersionedObjMetadata().from_dict(
            struct=struct, fields=fields, deprecated_fields=deprecated_fields
        )
        new_obj._status = ModelEndpointStatus().from_dict(
            struct=struct, fields=fields, deprecated_fields=deprecated_fields
        )
        new_obj._spec = ModelEndpointSpec().from_dict(
            struct=struct, fields=fields, deprecated_fields=deprecated_fields
        )
        return new_obj
