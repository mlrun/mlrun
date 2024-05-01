# Copyright 2024 Iguazio
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

import pytest

from mlrun.common.schemas.model_monitoring import (
    ModelEndpointMonitoringMetric,
    ModelEndpointMonitoringMetricType,
)
from mlrun.model_monitoring.db.stores.v3io_kv.kv_store import KVStoreBase


@pytest.mark.parametrize(
    ("items", "expected_metrics"),
    [
        ([], []),
        ([{"__name": ".#schema"}], []),
        (
            [
                {
                    "__name": "some-app-name",
                    "some-metric-name1": "string-attrs1",
                    "some-metric-name2": "string-attrs2",
                }
            ],
            [
                ModelEndpointMonitoringMetric(
                    app="some-app-name",
                    name="some-metric-name1",
                    type=ModelEndpointMonitoringMetricType.RESULT,
                    full_name="some-app-name/result/some-metric-name1",
                ),
                ModelEndpointMonitoringMetric(
                    app="some-app-name",
                    name="some-metric-name2",
                    type=ModelEndpointMonitoringMetricType.RESULT,
                    full_name="some-app-name/result/some-metric-name2",
                ),
            ],
        ),
        (
            [
                {
                    "__name": "app1",
                    "drift-res": "{'d':0.4}",
                },
                {"__name": ".#schema"},
            ],
            [
                ModelEndpointMonitoringMetric(
                    app="app1",
                    name="drift-res",
                    type=ModelEndpointMonitoringMetricType.RESULT,
                    full_name="app1/result/drift-res",
                )
            ],
        ),
    ],
)
def test_extract_metrics_from_items(
    items: list[dict[str, str]], expected_metrics: list[ModelEndpointMonitoringMetric]
) -> None:
    assert KVStoreBase._extract_metrics_from_items(items) == expected_metrics
