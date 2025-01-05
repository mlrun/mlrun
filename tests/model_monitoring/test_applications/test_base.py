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

from collections.abc import Iterator
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import Mock, patch

import pytest

import mlrun
from mlrun.common.schemas.model_monitoring import ResultKindApp, ResultStatusApp
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBase,
    ModelMonitoringApplicationResult,
    MonitoringApplicationContext,
)


class NoOpApp(ModelMonitoringApplicationBase):
    def do_tracking(self, monitoring_context: MonitoringApplicationContext):
        pass


class InProgressApp0(ModelMonitoringApplicationBase):
    def do_tracking(
        self, monitoring_context: MonitoringApplicationContext
    ) -> ModelMonitoringApplicationResult:
        monitoring_context.logger.info(
            "This test app is failing on purpose - ignore the failure!",
            project=monitoring_context.project_name,
        )
        raise ValueError


class InProgressApp1(ModelMonitoringApplicationBase):
    def do_tracking(
        self, monitoring_context: MonitoringApplicationContext
    ) -> ModelMonitoringApplicationResult:
        monitoring_context.logger.info(
            "It should work now",
            project=monitoring_context.project_name,
        )
        return ModelMonitoringApplicationResult(
            name="res0",
            value=0,
            status=ResultStatusApp.irrelevant,
            kind=ResultKindApp.mm_app_anomaly,
        )


@pytest.mark.filterwarnings("error")
def test_no_deprecation_instantiation() -> None:
    NoOpApp()


class TestEvaluate:
    @classmethod
    @pytest.fixture(autouse=True)
    def _set_project(cls) -> Iterator[None]:
        project = mlrun.get_or_create_project("test")
        with patch("mlrun.db.nopdb.NopDB.get_project", Mock(return_value=project)):
            yield

    @staticmethod
    def test_local_no_params() -> None:
        func_name = "test-app"
        run = InProgressApp0.evaluate(func_path=__file__, func_name=func_name)
        assert run.state() == "created"  # Should be "error", see ML-8507
        run = InProgressApp1.evaluate(func_path=__file__, func_name=func_name)
        assert run.state() == "completed"


@pytest.mark.parametrize(
    ("start", "end", "base_period", "expectation"),
    [
        (
            None,
            None,
            None,
            pytest.raises(
                mlrun.errors.MLRunValueError,
                match=".* you must also pass the start and end times",
            ),
        ),
        (
            datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
            datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc),
            None,
            does_not_raise(),
        ),
        (
            datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
            datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc),
            0,
            pytest.raises(
                mlrun.errors.MLRunValueError,
                match="`base_period` must be a nonnegative integer .*",
            ),
        ),
    ],
)
def test_validate_times(
    start: Optional[datetime],
    end: Optional[datetime],
    base_period: Optional[int],
    expectation: AbstractContextManager,
) -> None:
    with expectation:
        ModelMonitoringApplicationBase._validate_times(start, end, base_period)


@pytest.mark.parametrize(
    ("start", "end", "base_period", "expected_windows"),
    [
        (
            datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
            datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc),
            None,
            [
                (
                    datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
                    datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc),
                ),
            ],
        ),
        (
            datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
            datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc),
            600,
            [
                (
                    datetime(2008, 9, 1, 10, 2, 1, tzinfo=timezone.utc),
                    datetime(2008, 9, 1, 20, 2, 1, tzinfo=timezone.utc),
                ),
                (
                    datetime(2008, 9, 1, 20, 2, 1, tzinfo=timezone.utc),
                    datetime(2008, 9, 2, 6, 2, 1, tzinfo=timezone.utc),
                ),
                (
                    datetime(2008, 9, 2, 6, 2, 1, tzinfo=timezone.utc),
                    datetime(2008, 9, 2, 10, 2, 1, tzinfo=timezone.utc),
                ),
            ],
        ),
        (
            datetime(2024, 12, 26, 14, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 12, 26, 14, 4, 0, tzinfo=timezone.utc),
            1,
            [
                (
                    datetime(2024, 12, 26, 14, 0, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 26, 14, 1, 0, tzinfo=timezone.utc),
                ),
                (
                    datetime(2024, 12, 26, 14, 1, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 26, 14, 2, 0, tzinfo=timezone.utc),
                ),
                (
                    datetime(2024, 12, 26, 14, 2, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 26, 14, 3, 0, tzinfo=timezone.utc),
                ),
                (
                    datetime(2024, 12, 26, 14, 3, 0, tzinfo=timezone.utc),
                    datetime(2024, 12, 26, 14, 4, 0, tzinfo=timezone.utc),
                ),
            ],
        ),
    ],
)
def test_windows(
    start: datetime,
    end: datetime,
    base_period: Optional[int],
    expected_windows: list[tuple[datetime, datetime]],
) -> None:
    assert (
        list(
            ModelMonitoringApplicationBase._window_generator(
                start=start, end=end, base_period=base_period
            )
        )
        == expected_windows
    ), "The generated windows are different than expected"
