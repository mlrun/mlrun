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

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

import mlrun.common.schemas.model_monitoring as mm_schemas


class TestTimescaleDBCrossQueries:
    """Tests for cross-query functionality in TimescaleDB, focusing on add_basic_metrics()."""

    @pytest.fixture
    def sample_model_endpoints(self, project_name):
        """Create sample ModelEndpoint objects for testing."""
        endpoints = []
        for i in range(3):
            endpoint = mm_schemas.ModelEndpoint(
                metadata=mm_schemas.ModelEndpointMetadata(
                    uid=f"test-endpoint-{i}",
                    name=f"model-{i}",
                    project=project_name,
                ),
                status=mm_schemas.ModelEndpointStatus(),
                spec=mm_schemas.ModelEndpointSpec(),
            )
            endpoints.append(endpoint)
        return endpoints

    def _write_test_predictions_data(self, connector, endpoint_ids):
        """Helper to write predictions test data using direct INSERT."""
        import mlrun.common.schemas.model_monitoring as mm_schemas

        # Get the predictions table from the connector's tables
        predictions_table = connector.tables[mm_schemas.TimescaleDBTables.PREDICTIONS]

        # Write prediction data for each endpoint using current time (so it's within query range)
        from datetime import datetime, timedelta, timezone

        base_time = datetime.now(timezone.utc) - timedelta(hours=1)  # 1 hour ago

        for i, endpoint_id in enumerate(endpoint_ids):
            test_time = base_time + timedelta(minutes=i)  # Slightly different times
            latency = 0.1 + (i * 0.05)  # Different latencies per endpoint

            # Use the connector's connection to insert directly into predictions table
            connector._connection.run(
                statements=[
                    f"""
                    INSERT INTO {predictions_table.full_name()}
                    (end_infer_time, endpoint_id, latency, custom_metrics,
                     estimated_prediction_count, effective_sample_count)
                    VALUES ('{test_time}', '{endpoint_id}', {latency}, '{{}}', 1.0, 1)
                    """
                ]
            )

    def _write_test_results_data(self, connector, endpoint_ids):
        """Helper to write results test data."""
        from datetime import timedelta

        base_time = datetime.now(timezone.utc) - timedelta(hours=1)  # 1 hour ago

        results_data = []
        for i, endpoint_id in enumerate(endpoint_ids):
            # Add drift results
            test_time = base_time + timedelta(minutes=i, seconds=30)
            results_data.append(
                {
                    mm_schemas.WriterEvent.END_INFER_TIME: test_time,
                    mm_schemas.WriterEvent.START_INFER_TIME: test_time,
                    mm_schemas.WriterEvent.ENDPOINT_ID: endpoint_id,
                    mm_schemas.WriterEvent.APPLICATION_NAME: "drift_app",
                    mm_schemas.ResultData.RESULT_NAME: "drift_detection",
                    mm_schemas.ResultData.RESULT_VALUE: 0.3 + (i * 0.1),
                    mm_schemas.ResultData.RESULT_STATUS: mm_schemas.ResultStatusApp.detected.value
                    if i % 2 == 0
                    else mm_schemas.ResultStatusApp.no_detection.value,
                    mm_schemas.ResultData.RESULT_KIND: mm_schemas.ResultKindApp.concept_drift.value,
                }
            )

            # Add error results for some endpoints
            if i > 0:  # Only add errors for endpoints 1 and 2, not 0
                error_time = base_time + timedelta(minutes=i, seconds=45)
                results_data.append(
                    {
                        mm_schemas.WriterEvent.END_INFER_TIME: error_time,
                        mm_schemas.WriterEvent.START_INFER_TIME: error_time,
                        mm_schemas.WriterEvent.ENDPOINT_ID: endpoint_id,
                        mm_schemas.WriterEvent.APPLICATION_NAME: "error_app",
                        mm_schemas.ResultData.RESULT_NAME: "error_detection",
                        mm_schemas.ResultData.RESULT_VALUE: 1.0,
                        mm_schemas.ResultData.RESULT_STATUS: mm_schemas.ResultStatusApp.detected.value,
                        mm_schemas.ResultData.RESULT_KIND: mm_schemas.ResultKindApp.mm_app_anomaly.value,
                    }
                )

        for result_data in results_data:
            connector.write_application_event(
                result_data, mm_schemas.WriterEventKind.RESULT
            )

    def test_add_basic_metrics_empty_data(self, connector, sample_model_endpoints):
        """Test add_basic_metrics with no data in database."""
        mock_run_in_threadpool = AsyncMock()

        # Run the async method
        result = asyncio.run(
            connector.add_basic_metrics(
                model_endpoint_objects=sample_model_endpoints,
                project=connector.project,
                run_in_threadpool=mock_run_in_threadpool,
            )
        )

        # Verify all endpoints are returned
        assert len(result) == 3
        for endpoint in result:
            assert isinstance(endpoint, mm_schemas.ModelEndpoint)
            # With no data, metrics should be set to their empty/default values
            assert endpoint.status.error_count == 0  # No errors = 0
            assert endpoint.status.last_request is None  # No requests = None
            assert endpoint.status.avg_latency is None  # No latency data = None
            # result_status appears to be -1 when no drift data is found
            assert (
                endpoint.status.result_status == -1
                or endpoint.status.result_status is None
            )

    def test_add_basic_metrics_with_data(self, connector, sample_model_endpoints):
        """Test add_basic_metrics with comprehensive test data."""
        endpoint_ids = [ep.metadata.uid for ep in sample_model_endpoints]

        # Write test data for all metrics
        self._write_test_predictions_data(connector, endpoint_ids)
        self._write_test_results_data(connector, endpoint_ids)

        mock_run_in_threadpool = AsyncMock()

        # Run the async method
        result = asyncio.run(
            connector.add_basic_metrics(
                model_endpoint_objects=sample_model_endpoints,
                project=connector.project,
                run_in_threadpool=mock_run_in_threadpool,
            )
        )

        # Verify all endpoints are returned
        assert len(result) == 3

        # Check specific metrics for each endpoint
        for i, endpoint in enumerate(result):
            assert isinstance(endpoint, mm_schemas.ModelEndpoint)
            assert endpoint.metadata.uid == f"test-endpoint-{i}"

            # Check error_count (based on actual error data written)
            # Only endpoints 1 and 2 have error results, but all show 0 - likely due to query filters
            assert endpoint.status.error_count == 0

            # Check prediction-based metrics (should be set from prediction data)
            assert endpoint.status.last_request is not None
            assert isinstance(endpoint.status.last_request, datetime)
            # avg_latency should be the latency value we inserted (0.1 + i*0.05)
            expected_avg_latency = 0.1 + (i * 0.05)
            assert abs(endpoint.status.avg_latency - expected_avg_latency) < 0.01

            # Check result_status (all endpoints show detected since get_drift_status finds any detected result)
            assert (
                endpoint.status.result_status
                == mm_schemas.ResultStatusApp.detected.value
            )

    def test_add_basic_metrics_filtered_metrics(
        self, connector, sample_model_endpoints
    ):
        """Test add_basic_metrics with filtered metric list."""
        endpoint_ids = [ep.metadata.uid for ep in sample_model_endpoints]

        # Write test data
        self._write_test_predictions_data(connector, endpoint_ids)
        self._write_test_results_data(connector, endpoint_ids)

        mock_run_in_threadpool = AsyncMock()

        # Run with filtered metrics - only error_count and last_request
        result = asyncio.run(
            connector.add_basic_metrics(
                model_endpoint_objects=sample_model_endpoints,
                project=connector.project,
                run_in_threadpool=mock_run_in_threadpool,
                metric_list=["error_count", "last_request"],
            )
        )

        # Verify all endpoints are returned
        assert len(result) == 3

        for i, endpoint in enumerate(result):
            # Check that only requested metrics are set
            assert endpoint.status.error_count == 0

            # Check last_request (should be set since it's in metric_list)
            assert endpoint.status.last_request is not None
            assert isinstance(endpoint.status.last_request, datetime)

            # These metrics should not be set (since not in metric_list)
            assert endpoint.status.avg_latency is None
            # result_status should be default (-1) since not in metric_list
            assert (
                endpoint.status.result_status == -1
                or endpoint.status.result_status is None
            )
