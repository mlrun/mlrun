# Copyright 2026 Iguazio
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

import json
from datetime import datetime
from typing import Any, NamedTuple

from .constants import ModelEndpointMonitoringMetricType, ResultStatusApp


def compose_full_name(
    *,
    project: str,
    app: str,
    name: str,
    type: ModelEndpointMonitoringMetricType = ModelEndpointMonitoringMetricType.RESULT,
) -> str:
    return ".".join([project, app, type, name])


class MetricPoint(NamedTuple):
    timestamp: datetime
    value: float


class _ResultPoint(NamedTuple):
    timestamp: datetime
    value: float
    status: ResultStatusApp
    extra_data: str | None = ""


class _DriftBin(NamedTuple):
    timestamp: datetime
    count_suspected: int
    count_detected: int


def _json_loads_if_not_none(field: Any) -> Any:
    return (
        json.loads(field) if field and field != "null" and field is not None else None
    )
