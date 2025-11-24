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

from datetime import datetime, timezone
from typing import Optional

import pytest

import mlrun.common.schemas.model_monitoring as mm_schemas


class TestTimescaleDBCrossQueries:
    """Tests for cross-query functionality in TimescaleDB, focusing on add_basic_metrics()."""

    @staticmethod
    def _create_sample_model_endpoint(
        uid: str, name: str, project: str
    ) -> mm_schemas.ModelEndpoint:
        """Factory method for creating ModelEndpoint objects."""
        return mm_schemas.ModelEndpoint(
            metadata=mm_schemas.ModelEndpointMetadata(
                uid=uid,
                name=name,
                project=project,
            ),
            status=mm_schemas.ModelEndpointStatus(),
            spec=mm_schemas.ModelEndpointSpec(),
        )

    @staticmethod
    def _create_result_event_data(
        endpoint_id: str,
        application_name: str,
        result_name: str,
        result_value: float,
        result_status: int,
        result_kind: int,
        end_time: datetime,
        start_time: Optional[datetime] = None,
    ) -> dict:
        """Factory method for creating result event data."""
        if start_time is None:
            start_time = end_time

        return {
            mm_schemas.WriterEvent.END_INFER_TIME: end_time,
            mm_schemas.WriterEvent.START_INFER_TIME: start_time,
            mm_schemas.WriterEvent.ENDPOINT_ID: endpoint_id,
            mm_schemas.WriterEvent.APPLICATION_NAME: application_name,
            mm_schemas.ResultData.RESULT_NAME: result_name,
            mm_schemas.ResultData.RESULT_VALUE: result_value,
            mm_schemas.ResultData.RESULT_STATUS: result_status,
            mm_schemas.ResultData.RESULT_KIND: result_kind,
        }

    @staticmethod
    def _insert_prediction_data(
        connection, table, endpoint_id: str, end_time: datetime, latency: float
    ):
        """Helper method for inserting prediction data."""
        connection.run(
            statements=[
                f"""
                INSERT INTO {table.full_name()}
                (end_infer_time, endpoint_id, latency, custom_metrics,
                 estimated_prediction_count, effective_sample_count)
                VALUES ('{end_time}', '{endpoint_id}', {latency}, '{{}}', 1.0, 1)
                """
            ]
        )

    @staticmethod
    def _verify_basic_metrics(
        endpoints: list,
        expected_count: int,
        check_error_count: int = 0,
        check_last_request: bool = True,
        check_avg_latency: bool = True,
        check_result_status: Optional[int] = None,
    ):
        """Helper method for verifying basic metrics in endpoints."""
        assert len(endpoints) == expected_count

        for endpoint in endpoints:
            assert isinstance(endpoint, mm_schemas.ModelEndpoint)

            # Check error count
            assert endpoint.status.error_count == check_error_count

            # Check last_request
            if check_last_request:
                assert endpoint.status.last_request is not None
                assert isinstance(endpoint.status.last_request, datetime)
            else:
                assert endpoint.status.last_request is None

            # Check avg_latency
            if check_avg_latency:
                assert endpoint.status.avg_latency is not None
            else:
                assert endpoint.status.avg_latency is None

            # Check result_status
            if check_result_status is not None:
                assert endpoint.status.result_status == check_result_status

    @staticmethod
    def _verify_endpoint_specific_metrics(
        endpoints: list, expected_latencies: list, expected_result_status: int
    ):
        """Helper method for verifying endpoint-specific metrics."""
        for i, endpoint in enumerate(endpoints):
            assert endpoint.metadata.uid == f"test-endpoint-{i}"

            # Check avg_latency matches expected
            if expected_latencies:
                expected_avg_latency = expected_latencies[i]
                assert abs(endpoint.status.avg_latency - expected_avg_latency) < 0.01

            # Check result_status
            assert endpoint.status.result_status == expected_result_status

    @staticmethod
    def _assert_result_status_with_context(
        endpoint: mm_schemas.ModelEndpoint,
        expected_status: int,
        context_description: str,
    ):
        """Helper method for asserting result_status with clear context-specific error messages."""
        assert endpoint.status.result_status == expected_status, (
            f"{context_description}, result_status should be {expected_status}, "
            f"but got {endpoint.status.result_status} for endpoint {endpoint.metadata.uid}"
        )

    @pytest.fixture
    def sample_model_endpoints(self, project_name):
        """Create sample ModelEndpoint objects for testing."""
        return [
            self._create_sample_model_endpoint(
                f"test-endpoint-{i}", f"model-{i}", project_name
            )
            for i in range(3)
        ]

    def _write_test_predictions_data(self, connector, endpoint_ids):
        """Helper to write predictions test data using direct INSERT."""
        from datetime import datetime, timedelta, timezone

        predictions_table = connector._metrics_queries.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]
        base_time = datetime.now(timezone.utc) - timedelta(hours=1)  # 1 hour ago

        for i, endpoint_id in enumerate(endpoint_ids):
            test_time = base_time + timedelta(minutes=i)
            latency = 0.1 + (i * 0.05)
            self._insert_prediction_data(
                connector._connection,
                predictions_table,
                endpoint_id,
                test_time,
                latency,
            )

    def _write_test_results_data(self, connector, endpoint_ids):
        """Helper to write results test data using factory methods."""
        from datetime import timedelta

        base_time = datetime.now(timezone.utc) - timedelta(hours=1)  # 1 hour ago
        results_data = []

        for i, endpoint_id in enumerate(endpoint_ids):
            # Add drift results using factory method
            drift_time = base_time + timedelta(minutes=i, seconds=30)
            drift_status = (
                mm_schemas.ResultStatusApp.detected.value
                if i % 2 == 0
                else mm_schemas.ResultStatusApp.no_detection.value
            )
            results_data.append(
                self._create_result_event_data(
                    endpoint_id=endpoint_id,
                    application_name="drift_app",
                    result_name="drift_detection",
                    result_value=0.3 + (i * 0.1),
                    result_status=drift_status,
                    result_kind=mm_schemas.ResultKindApp.concept_drift.value,
                    end_time=drift_time,
                )
            )

            # Add error results for endpoints 1 and 2 using factory method
            if i > 0:
                error_time = base_time + timedelta(minutes=i, seconds=45)
                results_data.append(
                    self._create_result_event_data(
                        endpoint_id=endpoint_id,
                        application_name="error_app",
                        result_name="error_detection",
                        result_value=1.0,
                        result_status=mm_schemas.ResultStatusApp.detected.value,
                        result_kind=mm_schemas.ResultKindApp.mm_app_anomaly.value,
                        end_time=error_time,
                    )
                )

        # Write all result data
        for result_data in results_data:
            connector.write_application_event(
                result_data, mm_schemas.WriterEventKind.RESULT
            )

    def test_add_basic_metrics_empty_data(self, connector, sample_model_endpoints):
        """Test add_basic_metrics with no data in database."""

        # Run the async method
        result = connector.add_basic_metrics(
            model_endpoint_objects=sample_model_endpoints,
        )

        # Verify all endpoints are returned with empty data using helper
        self._verify_basic_metrics(
            endpoints=result,
            expected_count=3,
            check_error_count=0,
            check_last_request=False,  # No requests = None
            check_avg_latency=False,  # No latency data = None
        )
        # Verify result_status when no drift data exists in database
        for endpoint in result:
            assert endpoint.status.result_status == -1, (
                f"When no drift data exists, result_status should remain at default value -1, "
                f"but got {endpoint.status.result_status} for endpoint {endpoint.metadata.uid}"
            )

    def test_add_basic_metrics_with_data(self, connector, sample_model_endpoints):
        """Test add_basic_metrics with comprehensive test data."""
        endpoint_ids = [ep.metadata.uid for ep in sample_model_endpoints]

        # Write test data for all metrics
        self._write_test_predictions_data(connector, endpoint_ids)
        self._write_test_results_data(connector, endpoint_ids)

        # Run the async method
        result = connector.add_basic_metrics(
            model_endpoint_objects=sample_model_endpoints,
        )

        # Verify all endpoints are returned with data using helpers
        self._verify_basic_metrics(
            endpoints=result,
            expected_count=3,
            check_error_count=0,  # All show 0 due to query filters
            check_last_request=True,
            check_avg_latency=True,
        )

        # Check endpoint-specific metrics
        expected_latencies = [0.1 + (i * 0.05) for i in range(3)]
        self._verify_endpoint_specific_metrics(
            endpoints=result,
            expected_latencies=expected_latencies,
            expected_result_status=mm_schemas.ResultStatusApp.detected.value,
        )

    def test_add_basic_metrics_filtered_metrics(
        self, connector, sample_model_endpoints
    ):
        """Test add_basic_metrics with filtered metric list."""
        endpoint_ids = [ep.metadata.uid for ep in sample_model_endpoints]

        # Write test data
        self._write_test_predictions_data(connector, endpoint_ids)
        self._write_test_results_data(connector, endpoint_ids)

        # Run with filtered metrics - only error_count and last_request
        result = connector.add_basic_metrics(
            model_endpoint_objects=sample_model_endpoints,
            metric_list=["error_count", "last_request"],
        )

        # Verify filtered metrics using helper - only error_count and last_request should be set
        self._verify_basic_metrics(
            endpoints=result,
            expected_count=3,
            check_error_count=0,
            check_last_request=True,  # In metric_list
            check_avg_latency=False,  # Not in metric_list
        )

        # Verify result_status is not populated since 'result_status' not in metric_list
        for endpoint in result:
            assert endpoint.status.result_status == -1, (
                f"When 'result_status' not in metric_list, it should remain at default value -1, "
                f"but got {endpoint.status.result_status} for endpoint {endpoint.metadata.uid}"
            )
