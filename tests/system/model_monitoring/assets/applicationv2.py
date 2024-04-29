# Copyright 2023 Iguazio
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
import typing

import mlrun
import mlrun.common.model_monitoring.helpers
import mlrun.model_monitoring.applications.context as mm_context
import mlrun.model_monitoring.applications.results as mm_results
from mlrun.common.schemas.model_monitoring.constants import (
    ResultKindApp,
    ResultStatusApp,
)
from mlrun.model_monitoring.applications import (
    ModelMonitoringApplicationBaseV2,
    ModelMonitoringApplicationResult,
)

EXPECTED_EVENTS_COUNT = (
    mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events
)


class DemoMonitoringAppV2(ModelMonitoringApplicationBaseV2):
    NAME = "monitoring-test"
    check_num_events = True

    # noinspection PyMethodOverriding
    def __init_subclass__(cls, check_num_events: bool) -> None:
        super().__init_subclass__()
        cls.check_num_events = check_num_events

    def do_tracking(
        self,
        monitoring_context: mm_context.MonitoringApplicationContext,
    ) -> typing.Union[
        mm_results.ModelMonitoringApplicationResult,
        list[mm_results.ModelMonitoringApplicationResult],
    ]:
        monitoring_context.logger.info("Running demo app")
        if self.check_num_events:
            assert len(monitoring_context.sample_df) == EXPECTED_EVENTS_COUNT
        monitoring_context.logger.info("Asserted sample_df length")
        return [
            ModelMonitoringApplicationResult(
                name="data_drift_test",
                value=2.15,
                kind=ResultKindApp.data_drift,
                status=ResultStatusApp.detected,
            ),
            ModelMonitoringApplicationResult(
                name="model_perf",
                value=80,
                kind=ResultKindApp.model_performance,
                status=ResultStatusApp.no_detection,
            ),
        ]


class NoCheckDemoMonitoringApp(DemoMonitoringAppV2, check_num_events=False):
    pass
