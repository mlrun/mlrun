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

"""Tests for v2 DataFrame conversion helpers in TSDB base module (ML-11445)."""

import pandas as pd
import pytest

import mlrun.errors
from mlrun.common.schemas.model_monitoring import (
    AggregationConfig,
    ModelEndpointMonitoringMetric,
    ModelEndpointMonitoringMetricNoData,
    ModelEndpointMonitoringMetricValuesV2,
    ModelEndpointMonitoringResultValuesV2,
)
from mlrun.common.schemas.model_monitoring.constants import (
    MetricData,
    ModelEndpointMonitoringMetricType,
    ResultData,
    ResultKindApp,
    WriterEvent,
)
from mlrun.model_monitoring.db.tsdb.base import TSDBConnector


class TestDfToMetricsValuesV2:
    """Tests for df_to_metrics_values_v2 conversion helper."""

    def _create_metrics_df(self, data: list[dict]) -> pd.DataFrame:
        """Create a time-indexed DataFrame with metric data."""
        df = pd.DataFrame(data)
        df[WriterEvent.END_INFER_TIME] = pd.to_datetime(df[WriterEvent.END_INFER_TIME])
        df.set_index(WriterEvent.END_INFER_TIME, inplace=True)
        return df

    def _create_metric(
        self, project: str, app: str, name: str
    ) -> ModelEndpointMonitoringMetric:
        """Create a ModelEndpointMonitoringMetric."""
        return ModelEndpointMonitoringMetric(
            project=project,
            app=app,
            type=ModelEndpointMonitoringMetricType.METRIC,
            name=name,
        )

    def test_metrics_v2_raw_data(self):
        """Test conversion of raw (non-aggregated) metric data."""
        df = self._create_metrics_df(
            [
                {
                    WriterEvent.END_INFER_TIME: "2025-01-01 00:00:00",
                    WriterEvent.APPLICATION_NAME: "my-app",
                    MetricData.METRIC_NAME: "latency",
                    MetricData.METRIC_VALUE: 10.5,
                },
                {
                    WriterEvent.END_INFER_TIME: "2025-01-01 01:00:00",
                    WriterEvent.APPLICATION_NAME: "my-app",
                    MetricData.METRIC_NAME: "latency",
                    MetricData.METRIC_VALUE: 12.3,
                },
            ]
        )

        metrics = [self._create_metric("test-project", "my-app", "latency")]

        result = TSDBConnector.df_to_metrics_values_v2(
            df=df,
            metrics=metrics,
            project="test-project",
            agg_period=None,
            agg_functions=None,
        )

        assert len(result) == 1
        expected = ModelEndpointMonitoringMetricValuesV2(
            full_name="test-project.my-app.metric.latency",
            type=ModelEndpointMonitoringMetricType.METRIC,
            data=True,
            aggregation_config=AggregationConfig(
                aggregated=False, period=None, functions=None
            ),
            # Raw format: [timestamp, value]
            values=[
                [pd.Timestamp("2025-01-01 00:00:00"), 10.5],
                [pd.Timestamp("2025-01-01 01:00:00"), 12.3],
            ],
        )
        assert result[0] == expected

    def test_metrics_v2_aggregated_data(self):
        """Test conversion of aggregated metric data."""
        df = self._create_metrics_df(
            [
                {
                    WriterEvent.END_INFER_TIME: "2025-01-01 00:00:00",
                    WriterEvent.APPLICATION_NAME: "my-app",
                    MetricData.METRIC_NAME: "latency",
                    f"avg_{MetricData.METRIC_VALUE}": 10.0,
                    f"max_{MetricData.METRIC_VALUE}": 15.0,
                },
                {
                    WriterEvent.END_INFER_TIME: "2025-01-01 01:00:00",
                    WriterEvent.APPLICATION_NAME: "my-app",
                    MetricData.METRIC_NAME: "latency",
                    f"avg_{MetricData.METRIC_VALUE}": 12.0,
                    f"max_{MetricData.METRIC_VALUE}": 18.0,
                },
            ]
        )

        metrics = [self._create_metric("test-project", "my-app", "latency")]

        result = TSDBConnector.df_to_metrics_values_v2(
            df=df,
            metrics=metrics,
            project="test-project",
            agg_period="1h",
            agg_functions=["avg", "max"],
        )

        assert len(result) == 1
        expected = ModelEndpointMonitoringMetricValuesV2(
            full_name="test-project.my-app.metric.latency",
            type=ModelEndpointMonitoringMetricType.METRIC,
            data=True,
            aggregation_config=AggregationConfig(
                aggregated=True, period="1h", functions=["avg", "max"]
            ),
            # Aggregated format: [timestamp, avg, max]
            values=[
                [pd.Timestamp("2025-01-01 00:00:00"), 10.0, 15.0],
                [pd.Timestamp("2025-01-01 01:00:00"), 12.0, 18.0],
            ],
        )
        assert result[0] == expected

    def test_metrics_v2_empty_df(self):
        """Test conversion with empty DataFrame returns no-data objects."""
        df = pd.DataFrame()
        metrics = [self._create_metric("test-project", "my-app", "latency")]

        result = TSDBConnector.df_to_metrics_values_v2(
            df=df,
            metrics=metrics,
            project="test-project",
            agg_period=None,
            agg_functions=None,
        )

        assert len(result) == 1
        expected = ModelEndpointMonitoringMetricNoData(
            full_name="test-project.my-app.metric.latency",
            type=ModelEndpointMonitoringMetricType.METRIC,
            data=False,
        )
        assert result[0] == expected

    def test_metrics_v2_raises_error_when_aggregated_column_missing(self):
        """Test that an error is raised when expected aggregated column is missing."""
        # DataFrame with raw data but we claim it's aggregated
        df = self._create_metrics_df(
            [
                {
                    WriterEvent.END_INFER_TIME: "2025-01-01 00:00:00",
                    WriterEvent.APPLICATION_NAME: "my-app",
                    MetricData.METRIC_NAME: "latency",
                    MetricData.METRIC_VALUE: 10.5,  # raw column, not aggregated
                },
            ]
        )

        metrics = [self._create_metric("test-project", "my-app", "latency")]

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
            TSDBConnector.df_to_metrics_values_v2(
                df=df,
                metrics=metrics,
                project="test-project",
                agg_period="1h",
                agg_functions=["avg", "max"],
            )

        assert "avg_metric_value" in str(exc_info.value)
        assert "not found in DataFrame" in str(exc_info.value)


class TestDfToResultsValuesV2:
    """Tests for df_to_results_values_v2 conversion helper."""

    def _create_results_df(self, data: list[dict]) -> pd.DataFrame:
        """Create a time-indexed DataFrame with result data."""
        df = pd.DataFrame(data)
        df[WriterEvent.END_INFER_TIME] = pd.to_datetime(df[WriterEvent.END_INFER_TIME])
        df.set_index(WriterEvent.END_INFER_TIME, inplace=True)
        return df

    def _create_result_metric(
        self, project: str, app: str, name: str
    ) -> ModelEndpointMonitoringMetric:
        """Create a ModelEndpointMonitoringMetric for results."""
        return ModelEndpointMonitoringMetric(
            project=project,
            app=app,
            type=ModelEndpointMonitoringMetricType.RESULT,
            name=name,
            kind=ResultKindApp.data_drift,
        )

    def test_results_v2_raw_data(self):
        """Test conversion of raw (non-aggregated) result data with status and extra_data."""
        df = self._create_results_df(
            [
                {
                    WriterEvent.END_INFER_TIME: "2025-01-01 00:00:00",
                    WriterEvent.APPLICATION_NAME: "drift-app",
                    ResultData.RESULT_NAME: "general_drift",
                    ResultData.RESULT_VALUE: 0.15,
                    ResultData.RESULT_STATUS: 0,
                    ResultData.RESULT_EXTRA_DATA: '{"detail": "no drift"}',
                    ResultData.RESULT_KIND: ResultKindApp.data_drift.value,
                },
                {
                    WriterEvent.END_INFER_TIME: "2025-01-01 01:00:00",
                    WriterEvent.APPLICATION_NAME: "drift-app",
                    ResultData.RESULT_NAME: "general_drift",
                    ResultData.RESULT_VALUE: 0.85,
                    ResultData.RESULT_STATUS: 2,
                    ResultData.RESULT_EXTRA_DATA: '{"detail": "drift detected"}',
                    ResultData.RESULT_KIND: ResultKindApp.data_drift.value,
                },
            ]
        )

        metrics = [
            self._create_result_metric("test-project", "drift-app", "general_drift")
        ]

        result = TSDBConnector.df_to_results_values_v2(
            df=df,
            metrics=metrics,
            project="test-project",
            agg_period=None,
            agg_functions=None,
        )

        assert len(result) == 1
        expected = ModelEndpointMonitoringResultValuesV2(
            full_name="test-project.drift-app.result.general_drift",
            type=ModelEndpointMonitoringMetricType.RESULT,
            result_kind=ResultKindApp.data_drift,
            data=True,
            aggregation_config=AggregationConfig(
                aggregated=False, period=None, functions=None
            ),
            # Raw format: [timestamp, value, status, extra_data]
            values=[
                [
                    pd.Timestamp("2025-01-01 00:00:00"),
                    0.15,
                    0,
                    '{"detail": "no drift"}',
                ],
                [
                    pd.Timestamp("2025-01-01 01:00:00"),
                    0.85,
                    2,
                    '{"detail": "drift detected"}',
                ],
            ],
        )
        assert result[0] == expected

    def test_results_v2_aggregated_data_with_max_status(self):
        """Test that aggregated results include max_status but NOT extra_data."""
        df = self._create_results_df(
            [
                {
                    WriterEvent.END_INFER_TIME: "2025-01-01 00:00:00",
                    WriterEvent.APPLICATION_NAME: "drift-app",
                    ResultData.RESULT_NAME: "general_drift",
                    f"avg_{ResultData.RESULT_VALUE}": 0.20,
                    f"max_{ResultData.RESULT_VALUE}": 0.35,
                    f"max_{ResultData.RESULT_STATUS}": 0,  # max status in period
                    ResultData.RESULT_KIND: ResultKindApp.data_drift.value,
                },
                {
                    WriterEvent.END_INFER_TIME: "2025-01-01 01:00:00",
                    WriterEvent.APPLICATION_NAME: "drift-app",
                    ResultData.RESULT_NAME: "general_drift",
                    f"avg_{ResultData.RESULT_VALUE}": 0.50,
                    f"max_{ResultData.RESULT_VALUE}": 0.85,
                    f"max_{ResultData.RESULT_STATUS}": 2,  # detection in this period
                    ResultData.RESULT_KIND: ResultKindApp.data_drift.value,
                },
            ]
        )

        metrics = [
            self._create_result_metric("test-project", "drift-app", "general_drift")
        ]

        result = TSDBConnector.df_to_results_values_v2(
            df=df,
            metrics=metrics,
            project="test-project",
            agg_period="1h",
            agg_functions=["avg", "max"],
        )

        assert len(result) == 1
        expected = ModelEndpointMonitoringResultValuesV2(
            full_name="test-project.drift-app.result.general_drift",
            type=ModelEndpointMonitoringMetricType.RESULT,
            result_kind=ResultKindApp.data_drift,
            data=True,
            aggregation_config=AggregationConfig(
                aggregated=True, period="1h", functions=["avg", "max"]
            ),
            # Aggregated format: [timestamp, avg, max, max_status] - NO extra_data
            values=[
                [pd.Timestamp("2025-01-01 00:00:00"), 0.20, 0.35, 0],
                [pd.Timestamp("2025-01-01 01:00:00"), 0.50, 0.85, 2],
            ],
        )
        assert result[0] == expected

    def test_results_v2_empty_df(self):
        """Test conversion with empty DataFrame returns no-data objects."""
        df = pd.DataFrame()
        metrics = [
            self._create_result_metric("test-project", "drift-app", "general_drift")
        ]

        result = TSDBConnector.df_to_results_values_v2(
            df=df,
            metrics=metrics,
            project="test-project",
            agg_period=None,
            agg_functions=None,
        )

        assert len(result) == 1
        expected = ModelEndpointMonitoringMetricNoData(
            full_name="test-project.drift-app.result.general_drift",
            type=ModelEndpointMonitoringMetricType.RESULT,
            data=False,
        )
        assert result[0] == expected

    def test_results_v2_raises_error_when_aggregated_column_missing(self):
        """Test that an error is raised when expected aggregated column is missing."""
        # DataFrame with raw data but we claim it's aggregated
        df = self._create_results_df(
            [
                {
                    WriterEvent.END_INFER_TIME: "2025-01-01 00:00:00",
                    WriterEvent.APPLICATION_NAME: "drift-app",
                    ResultData.RESULT_NAME: "general_drift",
                    ResultData.RESULT_VALUE: 0.15,  # raw column, not aggregated
                    ResultData.RESULT_STATUS: 0,
                    ResultData.RESULT_EXTRA_DATA: "",
                    ResultData.RESULT_KIND: ResultKindApp.data_drift.value,
                },
            ]
        )

        metrics = [
            self._create_result_metric("test-project", "drift-app", "general_drift")
        ]

        with pytest.raises(mlrun.errors.MLRunInvalidArgumentError) as exc_info:
            TSDBConnector.df_to_results_values_v2(
                df=df,
                metrics=metrics,
                project="test-project",
                agg_period="1h",
                agg_functions=["avg", "max"],
            )

        assert "avg_result_value" in str(exc_info.value)
        assert "not found in DataFrame" in str(exc_info.value)
