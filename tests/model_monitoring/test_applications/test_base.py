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
            "This test app is failing on purpose - ignore the failure!"
        )
        raise ValueError


class InProgressApp1(ModelMonitoringApplicationBase):
    def do_tracking(
        self, monitoring_context: MonitoringApplicationContext
    ) -> ModelMonitoringApplicationResult:
        monitoring_context.logger.info("It should work now")
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
    def _set_project(cls) -> None:
        mlrun.get_or_create_project("test")

    @staticmethod
    def test_local_no_params() -> None:
        func_name = "test-app"
        run = InProgressApp0.evaluate(func_path=__file__, func_name=func_name)
        assert run.state() == "created"  # Should be "error", see ML-8507
        run = InProgressApp1.evaluate(func_path=__file__, func_name=func_name)
        assert run.state() == "completed"
