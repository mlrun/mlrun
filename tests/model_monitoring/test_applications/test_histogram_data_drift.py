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

import inspect
import logging
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

import mlrun.artifacts.manager
import mlrun.common.model_monitoring.helpers
import mlrun.model_monitoring.applications
import mlrun.model_monitoring.applications.context as mm_context
import mlrun.utils
from mlrun.common.schemas.model_monitoring.constants import (
    ResultKindApp,
    ResultStatusApp,
)
from mlrun.model_monitoring.applications.histogram_data_drift import (
    DataDriftClassifier,
    HistogramDataDriftApplication,
    InvalidMetricValueError,
    InvalidThresholdValueError,
)
from mlrun.utils import Logger

assets_folder = Path(__file__).parent / "assets"


@pytest.fixture
def application() -> HistogramDataDriftApplication:
    app = HistogramDataDriftApplication()
    return app


@pytest.fixture
def logger() -> mlrun.utils.Logger:
    return mlrun.utils.Logger(level=logging.DEBUG, name=__name__)


@pytest.fixture
def monitoring_context() -> Mock:
    mock_monitoring_context = Mock(spec=mm_context.MonitoringApplicationContext)
    mock_monitoring_context.log_stream = Logger(
        name="test_data_drift_app", level=logging.DEBUG
    )
    mock_monitoring_context._artifacts_manager = Mock(
        spec=mlrun.artifacts.manager.ArtifactManager
    )
    return mock_monitoring_context


class TestDataDriftClassifier:
    @staticmethod
    @pytest.mark.parametrize(
        ("potential", "detected"), [(0.4, 0.2), (0.0, 0.5), (0.7, 1.0), (-1, 2)]
    )
    def test_invalid_threshold(potential: float, detected: float) -> None:
        with pytest.raises(InvalidThresholdValueError):
            DataDriftClassifier(potential=potential, detected=detected)

    @staticmethod
    @given(
        st.one_of(
            st.floats(max_value=0, exclude_max=True),
            st.floats(min_value=1, exclude_min=True),
        )
    )
    def test_invalid_metric(value: float) -> None:
        with pytest.raises(InvalidMetricValueError):
            DataDriftClassifier().value_to_status(value)

    @staticmethod
    @pytest.fixture
    def classifier() -> DataDriftClassifier:
        return DataDriftClassifier(potential=0.5, detected=0.7)

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "expected_status"),
        [
            (0, ResultStatusApp.no_detection),
            (0.2, ResultStatusApp.no_detection),
            (0.5, ResultStatusApp.potential_detection),
            (0.6, ResultStatusApp.potential_detection),
            (0.71, ResultStatusApp.detected),
            (1, ResultStatusApp.detected),
        ],
    )
    def test_status(
        classifier: DataDriftClassifier, value: float, expected_status: ResultStatusApp
    ) -> None:
        assert (
            classifier.value_to_status(value) == expected_status
        ), "The status is different than expected"


class TestApplication:
    COUNT = 12  # the sample df size

    @classmethod
    @pytest.fixture
    def sample_df_stats(cls) -> mlrun.common.model_monitoring.helpers.FeatureStats:
        return mlrun.common.model_monitoring.helpers.FeatureStats(
            {
                "timestamp": {
                    "count": cls.COUNT,
                    "25%": "2024-03-11 09:31:39.152301+00:00",
                    "50%": "2024-03-11 09:31:39.152301+00:00",
                    "75%": "2024-03-11 09:31:39.152301+00:00",
                    "max": "2024-03-11 09:31:39.152301+00:00",
                    "mean": "2024-03-11 09:31:39.152301+00:00",
                    "min": "2024-03-11 09:31:39.152301+00:00",
                },
                "ticker": {
                    "count": cls.COUNT,
                    "unique": 1,
                    "top": "AAPL",
                    "freq": cls.COUNT,
                },
                "f1": {
                    "count": cls.COUNT,
                    "hist": [[2, 3, 0, 3, 1, 3], [-10, -5, 0, 5, 10, 15, 20]],
                },
                "f2": {
                    "count": cls.COUNT,
                    "hist": [[0, 6, 0, 2, 1, 3], [66, 67, 68, 69, 70, 71, 72]],
                },
                "l": {
                    "count": cls.COUNT,
                    "hist": [
                        [10, 0, 0, 0, 0, 2],
                        [0.0, 0.16, 0.33, 0.5, 0.67, 0.83, 1.0],
                    ],
                },
            }
        )

    @staticmethod
    @pytest.fixture
    def feature_stats() -> mlrun.common.model_monitoring.helpers.FeatureStats:
        return mlrun.common.model_monitoring.helpers.FeatureStats(
            {
                "f1": {
                    "count": 100,
                    "hist": [[0, 0, 0, 30, 70, 0], [-10, -5, 0, 5, 10, 15, 20]],
                },
                "f2": {
                    "count": 100,
                    "hist": [[0, 45, 5, 15, 35, 0], [66, 67, 68, 69, 70, 71, 72]],
                },
                "l": {
                    "count": 100,
                    "hist": [
                        [30, 0, 0, 0, 0, 70],
                        [0.0, 0.16, 0.33, 0.5, 0.67, 0.83, 1.0],
                    ],
                },
            }
        )

    @staticmethod
    @pytest.fixture
    def application_kwargs(
        sample_df_stats: mlrun.common.model_monitoring.helpers.FeatureStats,
        feature_stats: mlrun.common.model_monitoring.helpers.FeatureStats,
        application: HistogramDataDriftApplication,
        monitoring_context: Mock,
        logger: mlrun.utils.Logger,
    ) -> dict[str, Any]:
        kwargs = {}
        kwargs["monitoring_context"] = monitoring_context
        monitoring_context.application_name = application.NAME
        monitoring_context.sample_df_stats = sample_df_stats
        monitoring_context.feature_stats = feature_stats
        monitoring_context.sample_df = Mock(spec=pd.DataFrame)
        monitoring_context.start_infer_time = Mock(spec=pd.Timestamp)
        monitoring_context.end_infer_time = Mock(spec=pd.Timestamp)
        monitoring_context.latest_request = Mock(spec=pd.Timestamp)
        monitoring_context.endpoint_id = Mock(spec=str)
        monitoring_context.output_stream_uri = Mock(spec=str)
        monitoring_context.dict_to_histogram = (
            mm_context.MonitoringApplicationContext.dict_to_histogram
        )
        monitoring_context.logger = logger
        assert (
            kwargs.keys()
            == inspect.signature(application.do_tracking).parameters.keys()
        )
        return kwargs

    @classmethod
    def test(
        cls,
        application: HistogramDataDriftApplication,
        application_kwargs: dict[str, Any],
    ) -> None:
        results = application.do_tracking(**application_kwargs)
        metrics = []
        assert len(results) == 4, "Expected four results & metrics"
        for res in results:
            if isinstance(
                res,
                mlrun.model_monitoring.applications.ModelMonitoringApplicationResult,
            ):
                assert (
                    res.kind == ResultKindApp.data_drift
                ), "The kind should be data drift"
                assert (
                    res.name == "general_drift"
                ), "The result name should be general_drift"
                assert (
                    res.status == ResultStatusApp.potential_detection
                ), "Expected potential detection in the general drift"
            elif isinstance(
                res,
                mlrun.model_monitoring.applications.ModelMonitoringApplicationMetric,
            ):
                metrics.append(res)
        assert len(metrics) == 3, "Expected three metrics"


class TestMetricsPerFeature:
    @staticmethod
    @pytest.fixture
    def monitoring_context(
        logger: mlrun.utils.Logger,
    ) -> mm_context.MonitoringApplicationContext:
        ctx = Mock()

        def dict_to_histogram(df: pd.DataFrame) -> pd.DataFrame:
            return df

        ctx.dict_to_histogram = dict_to_histogram
        ctx.logger = logger
        return ctx

    @staticmethod
    @pytest.mark.parametrize(
        ("sample_df_stats", "feature_stats"),
        [
            pytest.param(pd.DataFrame(), pd.DataFrame(), id="empty-dfs"),
            pytest.param(
                pd.read_csv(assets_folder / "sample_df_stats.csv", index_col=0),
                pd.read_csv(assets_folder / "feature_stats.csv", index_col=0),
                id="real-world-csv-dfs",
            ),
        ],
    )
    def test_compute_metrics_per_feature(
        application: HistogramDataDriftApplication,
        monitoring_context: Mock,
        sample_df_stats: pd.DataFrame,
        feature_stats: pd.DataFrame,
    ) -> None:
        monitoring_context.sample_df_stats = sample_df_stats
        monitoring_context.feature_stats = feature_stats

        metrics_per_feature = application._compute_metrics_per_feature(
            monitoring_context=monitoring_context
        )
        assert set(metrics_per_feature.columns) == {
            metric.NAME for metric in application.metrics
        }, "Different metrics than expected"
        assert set(metrics_per_feature.index) == set(
            feature_stats.columns
        ), "The features are different than expected"
