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

from abc import ABC, abstractmethod
from typing import Union

from mlrun.serving.utils import StepToDict

from ..application import ModelMonitoringApplicationResult
from .context import MonitoringApplicationContext


class ModelMonitoringApplicationBaseV2(StepToDict, ABC):
    """
    A base class for a model monitoring application.
    Inherit from this class to create a custom model monitoring application.

    example for very simple custom application::
        # mlrun: start-code
        class MyApp(ApplicationBase):
            def do_tracking(
                self,
                sample_df_stats: mlrun.common.model_monitoring.helpers.FeatureStats,
                feature_stats: mlrun.common.model_monitoring.helpers.FeatureStats,
                start_infer_time: pd.Timestamp,
                end_infer_time: pd.Timestamp,
                schedule_time: pd.Timestamp,
                latest_request: pd.Timestamp,
                endpoint_id: str,
                output_stream_uri: str,
            ) -> ModelMonitoringApplicationResult:
                self.context.log_artifact(TableArtifact("sample_df_stats", df=self.dict_to_histogram(sample_df_stats)))
                return ModelMonitoringApplicationResult(
                    name="data_drift_test",
                    value=0.5,
                    kind=mm_constant.ResultKindApp.data_drift,
                    status=mm_constant.ResultStatusApp.detected,
                )

        # mlrun: end-code
    """

    kind = "monitoring_application"

    def do(
        self, monitoring_context: MonitoringApplicationContext
    ) -> tuple[list[ModelMonitoringApplicationResult], MonitoringApplicationContext]:
        """
        Process the monitoring event and return application results.

        :param monitoring_context:   (MonitoringApplicationContext) The monitoring context to process.
        :returns:                    (list[ModelMonitoringApplicationResult], dict) The application results
                                     and the original event for the application.
        """
        results = self.do_tracking(monitoring_context=monitoring_context)
        results = results if isinstance(results, list) else [results]
        return results, monitoring_context

    @abstractmethod
    def do_tracking(
        self,
        monitoring_context: MonitoringApplicationContext,
    ) -> Union[
        ModelMonitoringApplicationResult, list[ModelMonitoringApplicationResult]
    ]:
        """
        Implement this method with your custom monitoring logic.

        :param monitoring_context:      (MonitoringApplicationContext) The monitoring context to process.

        :returns:                       (ModelMonitoringApplicationResult) or
                                        (list[ModelMonitoringApplicationResult]) of the application results.
        """
        raise NotImplementedError
