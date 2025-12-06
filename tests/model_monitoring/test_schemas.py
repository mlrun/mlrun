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
    expected_result: Optional[ModelEndpointMonitoringMetricType],
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
            {"outer.inner_key": 99},
        ),
        # multiple nested levels
        (
            {"a": {"b": {"c-key": 5}}},
            {"a.b.c_key": 5},
        ),
        # mixed dicts and values
        (
            {"root": {"sub1": 1, "sub-2": {"deep-key": "x"}}, "plain": 7},
            {"root.sub1": 1, "root.sub_2.deep_key": "x", "plain": 7},
        ),
        # key with digit prefix deep inside
        (
            {"root": {"123abc": {"-bad-key": 1}}},
            {"root._123abc._bad_key": 1},
        ),
    ],
)
def test_normalize_dict(event, expected):
    result = _normalize_dict_for_v3io_frames(event)
    assert result == expected


def test_empty_dict():
    assert _normalize_dict_for_v3io_frames({}) == {}


# V2 API Schema Tests (ML-11445)


class TestAggregationConfig:
    """Tests for AggregationConfig schema."""

    def test_aggregation_config_aggregated_with_period_and_functions(self):
        """Test AggregationConfig with all fields populated."""
        from mlrun.common.schemas.model_monitoring import AggregationConfig

        config = AggregationConfig(
            aggregated=True,
            period="1h",
            functions=["avg", "min", "max"],
        )
        assert config.aggregated is True
        assert config.period == "1h"
        assert config.functions == ["avg", "min", "max"]

    def test_aggregation_config_not_aggregated(self):
        """Test AggregationConfig for raw (non-aggregated) data."""
        from mlrun.common.schemas.model_monitoring import AggregationConfig

        config = AggregationConfig(aggregated=False)
        assert config.aggregated is False
        assert config.period is None
        assert config.functions is None

    def test_aggregation_config_serialization_roundtrip(self):
        """Test AggregationConfig serialization and deserialization."""
        from mlrun.common.schemas.model_monitoring import AggregationConfig

        config = AggregationConfig(
            aggregated=True,
            period="6h",
            functions=["avg", "count"],
        )
        serialized = config.dict()
        deserialized = AggregationConfig(**serialized)
        assert deserialized.aggregated == config.aggregated
        assert deserialized.period == config.period
        assert deserialized.functions == config.functions


class TestModelEndpointMonitoringMetricValuesV2:
    """Tests for ModelEndpointMonitoringMetricValuesV2 schema."""

    def test_metric_values_v2_aggregated(self):
        """Test v2 metric values with aggregation."""
        from datetime import datetime

        from mlrun.common.schemas.model_monitoring import (
            AggregationConfig,
            ModelEndpointMonitoringMetricValuesV2,
        )
        from mlrun.common.schemas.model_monitoring.constants import (
            ModelEndpointMonitoringMetricType,
        )

        config = AggregationConfig(
            aggregated=True, period="1h", functions=["avg", "min", "max"]
        )
        values = ModelEndpointMonitoringMetricValuesV2(
            full_name="project.app.metric.latency",
            aggregation_config=config,
            values=[
                [datetime(2025, 1, 1, 0, 0), 10.5, 5.0, 20.0],
                [datetime(2025, 1, 1, 1, 0), 12.3, 6.0, 25.0],
            ],
        )
        assert values.full_name == "project.app.metric.latency"
        assert values.type == ModelEndpointMonitoringMetricType.METRIC
        assert values.data is True
        assert values.aggregation_config.aggregated is True
        assert len(values.values) == 2
        assert values.values[0][1] == 10.5  # avg
        assert values.values[0][2] == 5.0  # min
        assert values.values[0][3] == 20.0  # max

    def test_metric_values_v2_raw(self):
        """Test v2 metric values for raw (non-aggregated) data."""
        from datetime import datetime

        from mlrun.common.schemas.model_monitoring import (
            AggregationConfig,
            ModelEndpointMonitoringMetricValuesV2,
        )

        config = AggregationConfig(aggregated=False)
        values = ModelEndpointMonitoringMetricValuesV2(
            full_name="project.app.metric.error_count",
            aggregation_config=config,
            values=[
                [datetime(2025, 1, 1, 0, 0, 0), 1.0],
                [datetime(2025, 1, 1, 0, 0, 1), 2.0],
            ],
        )
        assert values.aggregation_config.aggregated is False
        assert len(values.values) == 2
        assert values.values[0][1] == 1.0


class TestModelEndpointMonitoringResultValuesV2:
    """Tests for ModelEndpointMonitoringResultValuesV2 schema."""

    def test_result_values_v2_aggregated(self):
        """Test v2 result values with aggregation.

        Note: Aggregated results only contain numeric aggregates (avg, max, etc.)
        without status or extra_data since those cannot be meaningfully aggregated.
        """
        from datetime import datetime

        from mlrun.common.schemas.model_monitoring import (
            AggregationConfig,
            ModelEndpointMonitoringResultValuesV2,
        )
        from mlrun.common.schemas.model_monitoring.constants import (
            ModelEndpointMonitoringMetricType,
            ResultKindApp,
        )

        config = AggregationConfig(
            aggregated=True, period="1h", functions=["avg", "max"]
        )
        # Aggregated values: [timestamp, avg_value, max_value] - no status/extra_data
        values = ModelEndpointMonitoringResultValuesV2(
            full_name="project.histogram-data-drift.result.general_drift",
            result_kind=ResultKindApp.data_drift,
            aggregation_config=config,
            values=[
                [datetime(2025, 1, 1, 0, 0), 0.15, 0.25],
                [datetime(2025, 1, 1, 1, 0), 0.18, 0.30],
            ],
        )
        assert values.full_name == "project.histogram-data-drift.result.general_drift"
        assert values.type == ModelEndpointMonitoringMetricType.RESULT
        assert values.result_kind == ResultKindApp.data_drift
        assert values.data is True
        assert values.aggregation_config.aggregated is True
        assert len(values.values) == 2
        # Each row has: timestamp + 2 aggregation values (avg, max)
        assert len(values.values[0]) == 3
        assert values.values[0][1] == 0.15  # avg
        assert values.values[0][2] == 0.25  # max

    def test_result_values_v2_raw(self):
        """Test v2 result values for raw (non-aggregated) data."""
        from datetime import datetime

        from mlrun.common.schemas.model_monitoring import (
            AggregationConfig,
            ModelEndpointMonitoringResultValuesV2,
        )
        from mlrun.common.schemas.model_monitoring.constants import ResultKindApp

        config = AggregationConfig(aggregated=False)
        values = ModelEndpointMonitoringResultValuesV2(
            full_name="project.app.result.score",
            result_kind=ResultKindApp.model_performance,
            aggregation_config=config,
            values=[
                [datetime(2025, 1, 1, 0, 0, 0), 0.95, 0, ""],
            ],
        )
        assert values.aggregation_config.aggregated is False
        assert values.result_kind == ResultKindApp.model_performance
        assert len(values.values) == 1
