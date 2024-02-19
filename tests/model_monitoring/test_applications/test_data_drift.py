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

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlrun.common.schemas.model_monitoring.constants import ResultStatusApp
from mlrun.model_monitoring.applications.data_drift import (
    DataDriftClassifier,
    InvalidMetricValueError,
    InvalidThresholdValueError,
)


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
