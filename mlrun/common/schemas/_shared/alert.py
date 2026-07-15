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

from mlrun.common.types import StrEnum


class EventEntityKind(StrEnum):
    MODEL_ENDPOINT_RESULT = "model-endpoint-result"
    MODEL_MONITORING_APPLICATION = "model-monitoring-application"
    MODEL_MONITORING_INFRA = "model-monitoring-infra"
    JOB = "job"


class EventKind(StrEnum):
    DATA_DRIFT_DETECTED = "data-drift-detected"
    DATA_DRIFT_SUSPECTED = "data-drift-suspected"
    CONCEPT_DRIFT_DETECTED = "concept-drift-detected"
    CONCEPT_DRIFT_SUSPECTED = "concept-drift-suspected"
    MODEL_PERFORMANCE_DETECTED = "model-performance-detected"
    MODEL_PERFORMANCE_SUSPECTED = "model-performance-suspected"
    SYSTEM_PERFORMANCE_DETECTED = "system-performance-detected"
    SYSTEM_PERFORMANCE_SUSPECTED = "system-performance-suspected"
    MM_APP_ANOMALY_DETECTED = "mm-app-anomaly-detected"
    MM_APP_ANOMALY_SUSPECTED = "mm-app-anomaly-suspected"
    MM_APP_FAILED = "mm-app-failed"
    MODEL_MONITORING_LAG_DETECTED = "model-monitoring-lag-detected"
    FAILED = "failed"


class AlertActiveState(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class AlertSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResetPolicy(StrEnum):
    MANUAL = "manual"
    AUTO = "auto"


class AlertsModes(StrEnum):
    enabled = "enabled"
    disabled = "disabled"


__all__ = [
    "AlertActiveState",
    "AlertSeverity",
    "AlertsModes",
    "EventEntityKind",
    "EventKind",
    "ResetPolicy",
]
