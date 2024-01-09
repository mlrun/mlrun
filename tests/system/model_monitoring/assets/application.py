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

import pandas as pd

import mlrun
from mlrun.common.schemas.model_monitoring.constants import (
    ResultKindApp,
    ResultStatusApp,
)
from mlrun.model_monitoring.application import (
    ModelMonitoringApplication,
    ApplicationResult,
)

EXPECTED_EVENTS_COUNT = (
    mlrun.mlconf.model_endpoint_monitoring.parquet_batching_max_events
)


class DemoMonitoringApp(ModelMonitoringApplication):
    name = "monitoring-test"
    check_num_events = True

    # noinspection PyMethodOverriding
    def __init_subclass__(cls, check_num_events: bool) -> None:
        super().__init_subclass__()
        cls.check_num_events = check_num_events

    def run_application(
        self,
        application_name: str,
        sample_df_stats: pd.DataFrame,
        feature_stats: pd.DataFrame,
        sample_df: pd.DataFrame,
        start_infer_time: pd.Timestamp,
        end_infer_time: pd.Timestamp,
        latest_request: pd.Timestamp,
        endpoint_id: str,
        output_stream_uri: str,
    ) -> list[ApplicationResult]:
        self.context.logger.info("Running demo app")
        if self.check_num_events:
            assert len(sample_df) == EXPECTED_EVENTS_COUNT
        self.context.logger.info("Asserted sample_df length")
        return [
            ApplicationResult(
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


class NoCheckDemoMonitoringApp(DemoMonitoringApp, check_num_events=False):
    pass
