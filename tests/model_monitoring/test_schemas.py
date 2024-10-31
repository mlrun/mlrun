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

import re
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any, Optional

import pydantic
import pytest

import mlrun.utils.regex
from mlrun.common.schemas.model_monitoring.constants import (
    PROJECT_PATTERN,
    ModelEndpointMonitoringMetricType,
)
from mlrun.common.schemas.model_monitoring.model_endpoints import (
    ModelEndpoint,
    ModelEndpointMonitoringMetric,
    _parse_metric_fqn_to_monitoring_metric,
)


@pytest.mark.parametrize(
    ("fqn", "expected_result", "expectation"),
    [
        (
            "1infer-model-tsdb-t3.histogram-data-drift.result.general_drift",
            ModelEndpointMonitoringMetric(
                project="1infer-model-tsdb-t3",
                app="histogram-data-drift",
                type=ModelEndpointMonitoringMetricType.RESULT,
                name="general_drift",
                full_name="1infer-model-tsdb-t3.histogram-data-drift.result.general_drift",
            ),
            does_not_raise(),
        ),
        (
            "proj_j.app-123.metric.error-count",
            ModelEndpointMonitoringMetric(
                project="proj_j",
                app="app-123",
                type=ModelEndpointMonitoringMetricType.METRIC,
                name="error-count",
                full_name="proj_j.app-123.metric.error-count",
            ),
            does_not_raise(),
        ),
        ("invalid..fqn", None, pytest.raises(ValueError)),
        ("prj.a.non-type.name", None, pytest.raises(ValueError)),
    ],
)
def test_fqn_parsing(
    fqn: str,
    expected_result: Optional[ModelEndpointMonitoringMetricType],
    expectation: AbstractContextManager,
) -> None:
    with expectation:
        assert _parse_metric_fqn_to_monitoring_metric(fqn) == expected_result


@pytest.mark.parametrize(
    ("flat_mep", "expectation"),
    [
        ({"project": "proj-1", "uid": "ok_30"}, does_not_raise()),
        (
            {},
            pytest.raises(
                pydantic.ValidationError,
                match=re.escape(
                    "2 validation errors for ModelEndpointMetadata\nproject\n  field required "
                    "(type=value_error.missing)\nuid\n  field required (type=value_error.missing)"
                ),
            ),
        ),
        (
            {"project": "im-fine-10"},
            pytest.raises(
                pydantic.ValidationError,
                match=(
                    re.escape(
                        "1 validation error for ModelEndpointMetadata\nuid\n  field required (type=value_error.missing)"
                    )
                ),
            ),
        ),
        (
            {"project": "im-fine-10", "uid": "xx' OR '1'='1"},
            pytest.raises(
                pydantic.ValidationError,
                match=(
                    re.escape(
                        "1 validation error for ModelEndpointMetadata\nuid\n  string does not match regex "
                        '"^[a-zA-Z0-9_-]+$" (type=value_error.str.regex; pattern=^[a-zA-Z0-9_-]+$)'
                    )
                ),
            ),
        ),
    ],
)
def test_model_endpoint_from_flat_dict(
    flat_mep: dict[str, Any], expectation: AbstractContextManager
) -> None:
    with expectation:
        ModelEndpoint.from_flat_dict(flat_mep)


def test_project_pattern() -> None:
    assert mlrun.utils.regex.project_name == [
        r"^.{0,63}$",
        r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$",
    ], f"The `project_name` regex changed, please update {PROJECT_PATTERN=} accordingly"
