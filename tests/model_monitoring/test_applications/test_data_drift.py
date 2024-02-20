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

from mlrun import MLClientCtx
from mlrun.common.schemas.model_monitoring.constants import ResultStatusApp
from mlrun.model_monitoring.applications.data_drift import (
    DataDriftClassifier,
    InvalidMetricValueError,
    InvalidThresholdValueError,
    MLRunDataDriftApplication,
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
    @pytest.fixture
    def classifier() -> DataDriftClassifier:
        return DataDriftClassifier(potential=0.5, detected=0.7)

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
    def test_status(value: float, expected_status: ResultStatusApp) -> None:
        assert (
            DataDriftClassifier(potential=0.5, detected=0.7).value_to_status(value)
            == expected_status
        ), "The status is different than expected"


class TestApplication:
    @staticmethod
    @pytest.fixture
    def sample_df_stats() -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {
                "f1": [0.1, 0.3, 0, 0.3, 0.05, 0.25],
                "f2": [0, 0.5, 0, 0.2, 0.05, 0.25],
                "l": [0.9, 0, 0, 0, 0, 0.1],
            }
        )

    @staticmethod
    @pytest.fixture
    def feature_stats() -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {
                "f1": [0, 0, 0, 0.3, 0.7, 0],
                "f2": [0, 0.45, 0.05, 0.15, 0.35, 0],
                "l": [0.3, 0, 0, 0, 0, 0.7],
            }
        )

    @staticmethod
    @pytest.fixture
    def application() -> MLRunDataDriftApplication:
        app = MLRunDataDriftApplication()
        app.context = MLClientCtx(
            log_stream=Logger(name="test_data_drift_app", level=logging.DEBUG)
        )
        return app

    @staticmethod
    @pytest.fixture
    def application_kwargs(
        sample_df_stats: pd.DataFrame,
        feature_stats: pd.DataFrame,
        application: MLRunDataDriftApplication,
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
        application: MLRunDataDriftApplication, application_kwargs: dict[str, Any]
    ) -> None:
        application.do_tracking(**application_kwargs)
