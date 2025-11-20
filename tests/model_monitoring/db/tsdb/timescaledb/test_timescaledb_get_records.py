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

import pandas as pd
import pytest

import mlrun.common.schemas.model_monitoring as mm_schemas


class TestGetRecords:
    """Tests for TimescaleDBConnector._get_records() method."""

    def test_get_records_metrics_all_endpoints(self, connector, query_test_helper):
        """Test _get_records() for metrics table with no endpoint filter."""
        # Insert test data for multiple endpoints
        test_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        test_data = [
            {
                mm_schemas.WriterEvent.END_INFER_TIME: test_time,
                mm_schemas.WriterEvent.START_INFER_TIME: test_time,
                mm_schemas.WriterEvent.ENDPOINT_ID: "endpoint-1",
                mm_schemas.WriterEvent.APPLICATION_NAME: "app1",
                mm_schemas.MetricData.METRIC_NAME: "accuracy",
                mm_schemas.MetricData.METRIC_VALUE: 0.95,
            },
            {
                mm_schemas.WriterEvent.END_INFER_TIME: test_time,
                mm_schemas.WriterEvent.START_INFER_TIME: test_time,
                mm_schemas.WriterEvent.ENDPOINT_ID: "endpoint-2",
                mm_schemas.WriterEvent.APPLICATION_NAME: "app2",
                mm_schemas.MetricData.METRIC_NAME: "precision",
                mm_schemas.MetricData.METRIC_VALUE: 0.87,
            },
        ]

        for data in test_data:
            query_test_helper.write_application_event(
                data, mm_schemas.WriterEventKind.METRIC
            )

        # Query all endpoints using _get_records
        df = connector._get_records(
            table="metrics",
            start=datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 16, 0, 0, 0, tzinfo=timezone.utc),
            endpoint_id=None,  # Get ALL endpoints
        )

        # Verify we got data for both endpoints
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == len(test_data)
        # Reference test_data instead of hardcoding expected values
        expected_apps = {d[mm_schemas.WriterEvent.APPLICATION_NAME] for d in test_data}
        expected_metrics = {d[mm_schemas.MetricData.METRIC_NAME] for d in test_data}
        assert set(df[mm_schemas.WriterEvent.APPLICATION_NAME]) == expected_apps
        assert set(df[mm_schemas.MetricData.METRIC_NAME]) == expected_metrics

    def test_get_records_metrics_specific_endpoint(self, connector, query_test_helper):
        """Test _get_records() for metrics table with endpoint filter."""
        # Insert test data for multiple endpoints
        test_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        test_data = [
            {
                mm_schemas.WriterEvent.END_INFER_TIME: test_time,
                mm_schemas.WriterEvent.START_INFER_TIME: test_time,
                mm_schemas.WriterEvent.ENDPOINT_ID: "endpoint-1",
                mm_schemas.WriterEvent.APPLICATION_NAME: "app1",
                mm_schemas.MetricData.METRIC_NAME: "accuracy",
                mm_schemas.MetricData.METRIC_VALUE: 0.95,
            },
            {
                mm_schemas.WriterEvent.END_INFER_TIME: test_time,
                mm_schemas.WriterEvent.START_INFER_TIME: test_time,
                mm_schemas.WriterEvent.ENDPOINT_ID: "endpoint-2",
                mm_schemas.WriterEvent.APPLICATION_NAME: "app2",
                mm_schemas.MetricData.METRIC_NAME: "precision",
                mm_schemas.MetricData.METRIC_VALUE: 0.87,
            },
        ]

        for data in test_data:
            query_test_helper.write_application_event(
                data, mm_schemas.WriterEventKind.METRIC
            )

        # Query specific endpoint using _get_records
        df = connector._get_records(
            table="metrics",
            start=datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 16, 0, 0, 0, tzinfo=timezone.utc),
            endpoint_id="endpoint-1",
        )

        # Verify we only got endpoint-1 (reference test_data)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == 1
        assert (
            df[mm_schemas.WriterEvent.APPLICATION_NAME].iloc[0]
            == test_data[0][mm_schemas.WriterEvent.APPLICATION_NAME]
        )
        assert (
            df[mm_schemas.MetricData.METRIC_NAME].iloc[0]
            == test_data[0][mm_schemas.MetricData.METRIC_NAME]
        )

    def test_get_records_results_all_endpoints(self, connector, query_test_helper):
        """Test _get_records() for results table with no endpoint filter."""
        # Insert test data for multiple endpoints
        test_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        test_data = [
            {
                mm_schemas.WriterEvent.END_INFER_TIME: test_time,
                mm_schemas.WriterEvent.START_INFER_TIME: test_time,
                mm_schemas.WriterEvent.ENDPOINT_ID: "endpoint-1",
                mm_schemas.WriterEvent.APPLICATION_NAME: "drift-app1",
                mm_schemas.ResultData.RESULT_NAME: "drift_detection",
                mm_schemas.ResultData.RESULT_VALUE: 10.0,
                mm_schemas.ResultData.RESULT_STATUS: mm_schemas.ResultStatusApp.detected.value,
                mm_schemas.ResultData.RESULT_KIND: mm_schemas.ResultKindApp.data_drift.value,
            },
            {
                mm_schemas.WriterEvent.END_INFER_TIME: test_time,
                mm_schemas.WriterEvent.START_INFER_TIME: test_time,
                mm_schemas.WriterEvent.ENDPOINT_ID: "endpoint-2",
                mm_schemas.WriterEvent.APPLICATION_NAME: "drift-app2",
                mm_schemas.ResultData.RESULT_NAME: "drift_detection",
                mm_schemas.ResultData.RESULT_VALUE: 20.0,
                mm_schemas.ResultData.RESULT_STATUS: mm_schemas.ResultStatusApp.detected.value,
                mm_schemas.ResultData.RESULT_KIND: mm_schemas.ResultKindApp.concept_drift.value,
            },
        ]

        for data in test_data:
            query_test_helper.write_application_event(
                data, mm_schemas.WriterEventKind.RESULT
            )

        # Query all endpoints using _get_records
        df = connector._get_records(
            table="results",
            start=datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 16, 0, 0, 0, tzinfo=timezone.utc),
            endpoint_id=None,  # Get ALL endpoints
        )

        # Verify we got data for both endpoints (reference test_data)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == len(test_data)
        expected_apps = {d[mm_schemas.WriterEvent.APPLICATION_NAME] for d in test_data}
        expected_values = {d[mm_schemas.ResultData.RESULT_VALUE] for d in test_data}
        assert set(df[mm_schemas.WriterEvent.APPLICATION_NAME]) == expected_apps
        assert set(df[mm_schemas.ResultData.RESULT_VALUE]) == expected_values

    def test_get_records_predictions_all_endpoints(self, connector):
        """Test _get_records() for predictions table with no endpoint filter."""
        # Insert test data for multiple endpoints directly into predictions table
        test_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        predictions_table = connector._metrics_queries.tables[
            mm_schemas.TimescaleDBTables.PREDICTIONS
        ]

        # Define test data
        test_data = [
            {"endpoint_id": "endpoint-1", "latency": 0.15},
            {"endpoint_id": "endpoint-2", "latency": 0.25},
        ]

        # Insert test data
        for data in test_data:
            connector._connection.run(
                statements=[
                    f"""
                    INSERT INTO {predictions_table.full_name()}
                    (end_infer_time, endpoint_id, latency, custom_metrics,
                     estimated_prediction_count, effective_sample_count)
                    VALUES ('{test_time}', '{data["endpoint_id"]}', {data["latency"]}, '{{}}', 1.0, 1)
                    """
                ]
            )

        # Query all endpoints using _get_records
        df = connector._get_records(
            table="predictions",
            start=datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 16, 0, 0, 0, tzinfo=timezone.utc),
            endpoint_id=None,  # Get ALL endpoints
        )

        # Verify we got data for both endpoints (reference test_data)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df) == len(test_data)
        expected_endpoints = {d["endpoint_id"] for d in test_data}
        expected_latencies = {d["latency"] for d in test_data}
        assert set(df[mm_schemas.WriterEvent.ENDPOINT_ID]) == expected_endpoints
        assert set(df["latency"]) == expected_latencies

    def test_get_records_empty_result(self, connector):
        """Test _get_records() returns empty DataFrame when no data exists."""
        df = connector._get_records(
            table="metrics",
            start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            endpoint_id=None,
        )

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_records_invalid_table(self, connector):
        """Test _get_records() raises error for invalid table name."""
        with pytest.raises(
            Exception,
            match="Invalid table.*Must be 'metrics', 'results', or 'predictions'",
        ):
            connector._get_records(
                table="invalid_table",
                start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                end=datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
            )

    def test_get_records_with_column_filter(self, connector, query_test_helper):
        """Test _get_records() with specific columns requested."""
        # Insert test data
        test_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        test_data = {
            mm_schemas.WriterEvent.END_INFER_TIME: test_time,
            mm_schemas.WriterEvent.START_INFER_TIME: test_time,
            mm_schemas.WriterEvent.ENDPOINT_ID: "endpoint-1",
            mm_schemas.WriterEvent.APPLICATION_NAME: "app1",
            mm_schemas.MetricData.METRIC_NAME: "accuracy",
            mm_schemas.MetricData.METRIC_VALUE: 0.95,
        }

        query_test_helper.write_application_event(
            test_data, mm_schemas.WriterEventKind.METRIC
        )

        # Query with specific columns
        df = connector._get_records(
            table="metrics",
            start=datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 16, 0, 0, 0, tzinfo=timezone.utc),
            endpoint_id="endpoint-1",
            columns=[
                mm_schemas.MetricData.METRIC_NAME,
                mm_schemas.MetricData.METRIC_VALUE,
            ],
        )

        # Verify only requested columns are returned
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert mm_schemas.MetricData.METRIC_NAME in df.columns
        assert mm_schemas.MetricData.METRIC_VALUE in df.columns
