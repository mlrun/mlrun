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
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

import mlrun.common.model_monitoring.helpers
from mlrun import MLClientCtx
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
    @staticmethod
    @pytest.fixture
    def sample_df_stats() -> mlrun.common.model_monitoring.helpers.FeatureStats:
        return mlrun.common.model_monitoring.helpers.FeatureStats(
            {
                "f1": {
                    "count": 100,
                    "hist": [[10, 30, 0, 30, 5, 25], [-10, -5, 0, 5, 10, 15, 20]],
                },
                "f2": {
                    "count": 100,
                    "hist": [[0, 50, 0, 20, 5, 25], [66, 67, 68, 69, 70, 71, 72]],
                },
                "l": {
                    "count": 100,
                    "hist": [
                        [90, 0, 0, 0, 0, 10],
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
    def application() -> HistogramDataDriftApplication:
        app = HistogramDataDriftApplication()
        app.context = MLClientCtx(
            log_stream=Logger(name="test_data_drift_app", level=logging.DEBUG)
        )
        return app

    @staticmethod
    @pytest.fixture
    def application_kwargs(
        sample_df_stats: mlrun.common.model_monitoring.helpers.FeatureStats,
        feature_stats: mlrun.common.model_monitoring.helpers.FeatureStats,
        application: HistogramDataDriftApplication,
    ) -> dict[str, Any]:
        kwargs = {}
        kwargs["application_name"] = application.NAME
        kwargs["sample_df_stats"] = sample_df_stats
        kwargs["feature_stats"] = feature_stats
        kwargs["sample_df"] = Mock(spec=pd.DataFrame)
        kwargs["start_infer_time"] = Mock(spec=pd.Timestamp)
        kwargs["end_infer_time"] = Mock(spec=pd.Timestamp)
        kwargs["latest_request"] = Mock(spec=pd.Timestamp)
        kwargs["endpoint_id"] = Mock(spec=str)
        kwargs["output_stream_uri"] = Mock(spec=str)
        assert (
            kwargs.keys()
            == inspect.signature(application.do_tracking).parameters.keys()
        )
        return kwargs

    @staticmethod
    def test(
        application: HistogramDataDriftApplication, application_kwargs: dict[str, Any]
    ) -> None:
        results = application.do_tracking(**application_kwargs)
        assert len(results) == 4, "Expected four results"
        result_by_name: dict[str, dict] = {}
        for res in results:
            assert res.kind == ResultKindApp.data_drift, "The kind should be data drift"
            res_dict = res.to_dict()
            result_by_name[res_dict.pop("result_name")] = res_dict

        assert (
            result_by_name["kld_mean"]["result_status"] == ResultStatusApp.irrelevant
        ), "KL divergence status is currently irrelevant"

        assert (
            result_by_name["general_drift"]["result_status"]
            == ResultStatusApp.potential_detection
        ), "Expected potential detection in the general drift"
