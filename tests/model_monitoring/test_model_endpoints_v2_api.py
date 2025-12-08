# Copyright 2025 Iguazio
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

"""
Tests for v2 model endpoints API parameter handling and aggregation logic.
"""

from datetime import UTC, datetime, timedelta

import mlrun.common.schemas.model_monitoring.constants as mm_constants
import mlrun.common.schemas.model_monitoring.model_endpoints as mm_endpoints


class TestV2ResultCountLimit:
    """Tests for v2 API result count validation."""

    def test_max_results_constant_is_500(self):
        """Test that the max results constant is set to 500."""
        from server.py.services.api.api.endpoints.model_endpoints_v2 import (
            _MAX_RESULTS_PER_METRIC,
        )

        assert _MAX_RESULTS_PER_METRIC == 500

    def test_result_count_validation_logic(self):
        """Test the result count validation logic."""
        from server.py.services.api.api.endpoints.model_endpoints_v2 import (
            _MAX_RESULTS_PER_METRIC,
        )

        # Simulate validation logic from the endpoint
        now = datetime.now(UTC)
        values_under_limit = [
            [now + timedelta(minutes=i), float(i)] for i in range(100)
        ]
        values_at_limit = [[now + timedelta(minutes=i), float(i)] for i in range(500)]
        values_over_limit = [[now + timedelta(minutes=i), float(i)] for i in range(501)]

        metric_under = mm_endpoints.ModelEndpointMonitoringMetricValuesV2(
            full_name="project.app.metric.test",
            aggregation_config=mm_endpoints.AggregationConfig(
                aggregated=False, period=None, functions=None
            ),
            values=values_under_limit,
        )
        metric_at = mm_endpoints.ModelEndpointMonitoringMetricValuesV2(
            full_name="project.app.metric.test",
            aggregation_config=mm_endpoints.AggregationConfig(
                aggregated=False, period=None, functions=None
            ),
            values=values_at_limit,
        )
        metric_over = mm_endpoints.ModelEndpointMonitoringMetricValuesV2(
            full_name="project.app.metric.test",
            aggregation_config=mm_endpoints.AggregationConfig(
                aggregated=False, period=None, functions=None
            ),
            values=values_over_limit,
        )

        # Under limit should pass
        assert len(metric_under.values) <= _MAX_RESULTS_PER_METRIC

        # At limit should pass
        assert len(metric_at.values) <= _MAX_RESULTS_PER_METRIC

        # Over limit should fail
        assert len(metric_over.values) > _MAX_RESULTS_PER_METRIC


class TestV2ResponseSchemas:
    """Tests for v2 response schema structure."""

    def test_metric_values_v2_has_aggregation_config(self):
        """Test that ModelEndpointMonitoringMetricValuesV2 has aggregation_config."""
        metric = mm_endpoints.ModelEndpointMonitoringMetricValuesV2(
            full_name="project.app.metric.metric_name",
            aggregation_config=mm_endpoints.AggregationConfig(
                aggregated=True,
                period="1h",
                functions=["avg"],
            ),
            values=[
                [datetime.now(UTC), 1.0],
            ],
        )
        assert metric.aggregation_config is not None
        assert metric.aggregation_config.aggregated is True
        assert metric.aggregation_config.period == "1h"
        assert metric.aggregation_config.functions == ["avg"]

    def test_result_values_v2_has_aggregation_config(self):
        """Test that ModelEndpointMonitoringResultValuesV2 has aggregation_config."""
        result = mm_endpoints.ModelEndpointMonitoringResultValuesV2(
            full_name="project.app.result_name",
            result_kind=mm_constants.ResultKindApp.data_drift,
            aggregation_config=mm_endpoints.AggregationConfig(
                aggregated=False,
                period=None,
                functions=None,
            ),
            values=[
                [datetime.now(UTC), 0.5, 0, ""],
            ],
        )
        assert result.aggregation_config is not None
        assert result.aggregation_config.aggregated is False


class TestV2MetricValuesFormat:
    """Tests for v2 metric values format."""

    def test_raw_metric_values_format(self):
        """Test that raw metric values have [timestamp, value] format."""
        now = datetime.now(UTC)
        values = [
            [now, 1.0],
            [now + timedelta(minutes=1), 2.0],
        ]
        metric = mm_endpoints.ModelEndpointMonitoringMetricValuesV2(
            full_name="project.app.metric.metric_name",
            aggregation_config=mm_endpoints.AggregationConfig(
                aggregated=False,
                period=None,
                functions=None,
            ),
            values=values,
        )
        assert len(metric.values) == 2
        assert len(metric.values[0]) == 2  # [timestamp, value]

    def test_aggregated_metric_values_format(self):
        """Test that aggregated metric values have [timestamp, agg1, agg2, ...] format."""
        now = datetime.now(UTC)
        # Format: [timestamp, avg_value, min_value, max_value]
        values = [
            [now, 1.5, 1.0, 2.0],
            [now + timedelta(hours=1), 2.5, 2.0, 3.0],
        ]
        metric = mm_endpoints.ModelEndpointMonitoringMetricValuesV2(
            full_name="project.app.metric.metric_name",
            aggregation_config=mm_endpoints.AggregationConfig(
                aggregated=True,
                period="1h",
                functions=["avg", "min", "max"],
            ),
            values=values,
        )
        assert len(metric.values) == 2
        # [timestamp, avg, min, max] = 4 elements
        assert len(metric.values[0]) == 4

    def test_raw_result_values_format(self):
        """Test that raw result values have [timestamp, value, status, extra_data] format."""
        now = datetime.now(UTC)
        values = [
            [now, 0.5, 0, ""],
            [now + timedelta(minutes=1), 0.8, 1, '{"info": "drift detected"}'],
        ]
        result = mm_endpoints.ModelEndpointMonitoringResultValuesV2(
            full_name="project.app.result_name",
            result_kind=mm_constants.ResultKindApp.data_drift,
            aggregation_config=mm_endpoints.AggregationConfig(
                aggregated=False,
                period=None,
                functions=None,
            ),
            values=values,
        )
        assert len(result.values) == 2
        assert len(result.values[0]) == 4  # [timestamp, value, status, extra_data]
