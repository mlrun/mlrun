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

from typing import Any, Dict, List, Optional

import mlrun.model
from mlrun.common.schemas.model_monitoring.constants import (
    EndpointType,
    EventKeyMetrics,
    EventLiveStats,
    ModelMonitoringMode,
)


class ModelEndpointSpec(mlrun.model.ModelObj):
    def __init__(
        self,
        function_uri: Optional[str] = "",
        model: Optional[str] = "",
        model_class: Optional[str] = "",
        model_uri: Optional[str] = "",
        feature_names: Optional[List[str]] = None,
        label_names: Optional[List[str]] = None,
        stream_path: Optional[str] = "",
        algorithm: Optional[str] = "",
        monitor_configuration: Optional[dict] = None,
        active: Optional[bool] = True,
        monitoring_mode: Optional[ModelMonitoringMode] = ModelMonitoringMode.disabled,
    ):
        self.function_uri = function_uri  # <project_name>/<function_name>:<tag>
        self.model = model  # <model_name>:<version>
        self.model_class = model_class
        self.model_uri = model_uri
        self.feature_names = feature_names or []
        self.label_names = label_names or []
        self.stream_path = stream_path
        self.algorithm = algorithm
        self.monitor_configuration = monitor_configuration or {}
        self.active = active
        self.monitoring_mode = monitoring_mode


class ModelEndpointStatus(mlrun.model.ModelObj):
    def __init__(
        self,
        feature_stats: Optional[dict] = None,
        current_stats: Optional[dict] = None,
        first_request: Optional[str] = "",
        last_request: Optional[str] = "",
        error_count: Optional[int] = 0,
        drift_status: Optional[str] = "",
        drift_measures: Optional[dict] = None,
        metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        features: Optional[List[Dict[str, Any]]] = None,
        children: Optional[List[str]] = None,
        children_uids: Optional[List[str]] = None,
        endpoint_type: Optional[EndpointType] = EndpointType.NODE_EP.value,
        monitoring_feature_set_uri: Optional[str] = "",
        state: Optional[str] = "",
    ):
        self.feature_stats = feature_stats or {}
        self.current_stats = current_stats or {}
        self.first_request = first_request
        self.last_request = last_request
        self.error_count = error_count
        self.drift_status = drift_status
        self.drift_measures = drift_measures or {}
        self.features = features or []
        self.children = children or []
        self.children_uids = children_uids or []
        self.endpoint_type = endpoint_type
        self.monitoring_feature_set_uri = monitoring_feature_set_uri
        if metrics is None:
            self.metrics = {
                EventKeyMetrics.GENERIC: {
                    EventLiveStats.LATENCY_AVG_1H: 0,
                    EventLiveStats.PREDICTIONS_PER_SECOND: 0,
                }
            }
        self.state = state


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
