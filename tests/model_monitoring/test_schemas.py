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
from typing import Any

import pydantic.v1
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
from mlrun.model_monitoring.db.tsdb.v3io.stream_graph_steps import (
    _normalize_dict_for_v3io_frames,
)


@pytest.mark.parametrize(
    ("fqn", "expected_result", "expectation"),
    [
        (
            "infer-model-tsdb-t3.histogram-data-drift.result.general_drift",
            ModelEndpointMonitoringMetric(
                project="infer-model-tsdb-t3",
                app="histogram-data-drift",
                type=ModelEndpointMonitoringMetricType.RESULT,
                name="general_drift",
            ),
            does_not_raise(),
        ),
        (
            "proj-j.app-123.metric.error_count",
            ModelEndpointMonitoringMetric(
                project="proj-j",
                app="app-123",
                type=ModelEndpointMonitoringMetricType.METRIC,
                name="error_count",
            ),
            does_not_raise(),
        ),
        ("invalid..fqn", None, pytest.raises(ValueError)),
        ("prj.a.non-type.name", None, pytest.raises(ValueError)),
    ],
)
def test_fqn_parsing(
    fqn: str,
    expected_result: ModelEndpointMonitoringMetricType | None,
    expectation: AbstractContextManager,
) -> None:
    with expectation:
        assert _parse_metric_fqn_to_monitoring_metric(fqn) == expected_result


@pytest.mark.parametrize(
    ("flat_mep", "validate", "expectation"),
    [
        (
            {
                "project": "proj-1",
                "uid": "81d488cf-0104-4bb4-98c4-e4fd1204e82f",
                "name": "test",
            },
            True,
            does_not_raise(),
        ),
        ({}, True, pytest.raises(pydantic.v1.ValidationError)),
        (
            {"project": "im-fine-10"},
            True,
            pytest.raises(
                pydantic.v1.ValidationError,
                match=(
                    re.escape(
                        "1 validation error for ModelEndpointMetadata\nname\n  "
                        "field required (type=value_error.missing)"
                    )
                ),
            ),
        ),
        (
            {"project": "im-fine-10", "uid": "xx' OR '1'='1", "name": "test"},
            True,
            pytest.raises(
                pydantic.v1.ValidationError,
                match=re.escape(
                    "1 validation error for ModelEndpointMetadata\nuid\n  "
                    "string does not match regex "
                    '"^[a-zA-Z0-9_-]+$" (type=value_error.str.regex; pattern=^[a-zA-Z0-9_-]+$)'
                ),
            ),
        ),
        (
            {"project": "im-fine-10", "uid": "xx' OR '1'='1", "name": "test"},
            False,
            does_not_raise(),
        ),
    ],
)
def test_model_endpoint_from_flat_dict(
    flat_mep: dict[str, Any], validate: bool, expectation: AbstractContextManager
) -> None:
    with expectation:
        ModelEndpoint.from_flat_dict(flat_mep, validate=validate)


def test_project_pattern() -> None:
    assert mlrun.utils.regex.project_name == [
        r"^.{0,63}$",
        r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$",
    ], f"The `project_name` regex changed, please update {PROJECT_PATTERN=} accordingly"


@pytest.mark.parametrize(
    "event,expected",
    [
        # basic case: valid key
        ({"validKey": 1}, {"validKey": 1}),
        # hyphens replaced with underscores
        ({"key-name": 42}, {"key_name": 42}),
        # keys starting with digit
        ({"123abc": "value"}, {"_123abc": "value"}),
        # nested dict flattening
        (
            {"outer": {"inner-key": 99}},
            {"outer:inner_key": 99},
        ),
        # multiple nested levels
        (
            {"a": {"b": {"c-key": 5}}},
            {"a:b:c_key": 5},
        ),
        # mixed dicts and values
        (
            {"root": {"sub1": 1, "sub-2": {"deep-key": "x"}}, "plain": 7},
            {"root:sub1": 1, "root:sub_2:deep_key": "x", "plain": 7},
        ),
        # key with digit prefix deep inside
        (
            {"root": {"123abc": {"-bad-key": 1}}},
            {"root:_123abc:_bad_key": 1},
        ),
    ],
)
def test_normalize_dict(event, expected):
    result = _normalize_dict_for_v3io_frames(event)
    assert result == expected


def test_empty_dict():
    assert _normalize_dict_for_v3io_frames({}) == {}
