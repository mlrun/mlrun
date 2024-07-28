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
from typing import Any, Union, cast

import numpy as np
import pandas as pd

import mlrun
import mlrun.model_monitoring.applications.context as mm_context
import mlrun.model_monitoring.applications.results as mm_results
from mlrun.serving.utils import MonitoringApplicationToDict


class ModelMonitoringApplicationBaseV2(MonitoringApplicationToDict, ABC):
    """
    A base class for a model monitoring application.
    Inherit from this class to create a custom model monitoring application.

    example for very simple custom application::

        class MyApp(ApplicationBase):
            def do_tracking(
                self,
                monitoring_context: mm_context.MonitoringApplicationContext,
            ) -> ModelMonitoringApplicationResult:
                monitoring_context.log_artifact(
                    TableArtifact(
                        "sample_df_stats", df=self.dict_to_histogram(sample_df_stats)
                    )
                )
                return ModelMonitoringApplicationResult(
                    name="data_drift_test",
                    value=0.5,
                    kind=mm_constant.ResultKindApp.data_drift,
                    status=mm_constant.ResultStatusApp.detected,
                )


    """

    kind = "monitoring_application"

    def do(
        self, monitoring_context: mm_context.MonitoringApplicationContext
    ) -> tuple[
        list[
            Union[
                mm_results.ModelMonitoringApplicationResult,
                mm_results.ModelMonitoringApplicationMetric,
            ]
        ],
        mm_context.MonitoringApplicationContext,
    ]:
        """
        Process the monitoring event and return application results & metrics.

        :param monitoring_context:   (MonitoringApplicationContext) The monitoring application context.
        :returns:                    A tuple of:
                                        [0] = list of application results that can be either from type
                                        `ModelMonitoringApplicationResult`
                                        or from type `ModelMonitoringApplicationResult`.
                                        [1] = the original application event, wrapped in `MonitoringApplicationContext`
                                         object
        """
        results = self.do_tracking(monitoring_context=monitoring_context)
        if isinstance(results, dict):
            results = [
                mm_results.ModelMonitoringApplicationMetric(name=key, value=value)
                for key, value in results.items()
            ]
        results = results if isinstance(results, list) else [results]
        return results, monitoring_context

    @abstractmethod
    def do_tracking(
        self,
        monitoring_context: mm_context.MonitoringApplicationContext,
    ) -> Union[
        mm_results.ModelMonitoringApplicationResult,
        list[
            Union[
                mm_results.ModelMonitoringApplicationResult,
                mm_results.ModelMonitoringApplicationMetric,
            ]
        ],
        dict[str, Any],
    ]:
        """
        Implement this method with your custom monitoring logic.

        :param monitoring_context:      (MonitoringApplicationContext) The monitoring context to process.

        :returns:                       (ModelMonitoringApplicationResult) or
                                        (list[Union[ModelMonitoringApplicationResult,
                                        ModelMonitoringApplicationMetric]])
                                        or dict that contains the application metrics only (in this case the name of
                                        each metric name is the key and the metric value is the corresponding value).
        """
        raise NotImplementedError


class ModelMonitoringApplicationBase(MonitoringApplicationToDict, ABC):
    """
    A base class for a model monitoring application.
    Inherit from this class to create a custom model monitoring application.

    example for very simple custom application::

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
                self.context.log_artifact(
                    TableArtifact(
                        "sample_df_stats", df=self.dict_to_histogram(sample_df_stats)
                    )
                )
                return ModelMonitoringApplicationResult(
                    name="data_drift_test",
                    value=0.5,
                    kind=mm_constant.ResultKindApp.data_drift,
                    status=mm_constant.ResultStatusApp.detected,
                )


    """

    kind = "monitoring_application"

    def do(
        self, monitoring_context: mm_context.MonitoringApplicationContext
    ) -> tuple[
        list[mm_results.ModelMonitoringApplicationResult],
        mm_context.MonitoringApplicationContext,
    ]:
        """
        Process the monitoring event and return application results.

        :param monitoring_context:   (MonitoringApplicationContext) The monitoring context to process.
        :returns:                    A tuple of:
                                        [0] = list of application results that can be either from type
                                        `ModelMonitoringApplicationResult` or from type
                                        `ModelMonitoringApplicationResult`.
                                        [1] = the original application event, wrapped in `MonitoringApplicationContext`
                                         object
        """
        resolved_event = self._resolve_event(monitoring_context)
        if not (
            hasattr(self, "context") and isinstance(self.context, mlrun.MLClientCtx)
        ):
            self._lazy_init(monitoring_context)
        results = self.do_tracking(*resolved_event)
        results = results if isinstance(results, list) else [results]
        return results, monitoring_context

    def _lazy_init(self, monitoring_context: mm_context.MonitoringApplicationContext):
        self.context = cast(mlrun.MLClientCtx, monitoring_context)

    @abstractmethod
    def do_tracking(
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
    ) -> Union[
        mm_results.ModelMonitoringApplicationResult,
        list[mm_results.ModelMonitoringApplicationResult],
    ]:
        """
        Implement this method with your custom monitoring logic.

        :param application_name:        (str) the app name
        :param sample_df_stats:         (pd.DataFrame) The new sample distribution.
        :param feature_stats:           (pd.DataFrame) The train sample distribution.
        :param sample_df:               (pd.DataFrame) The new sample DataFrame.
        :param start_infer_time:        (pd.Timestamp) Start time of the monitoring schedule.
        :param end_infer_time:          (pd.Timestamp) End time of the monitoring schedule.
        :param latest_request:          (pd.Timestamp) Timestamp of the latest request on this endpoint_id.
        :param endpoint_id:             (str) ID of the monitored model endpoint
        :param output_stream_uri:       (str) URI of the output stream for results

        :returns:                       (ModelMonitoringApplicationResult) or
                                        (list[ModelMonitoringApplicationResult]) of the application results.
        """
        raise NotImplementedError

    @classmethod
    def _resolve_event(
        cls,
        monitoring_context: mm_context.MonitoringApplicationContext,
    ) -> tuple[
        str,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Timestamp,
        pd.Timestamp,
        pd.Timestamp,
        str,
        str,
    ]:
        """
        Converting the event into a single tuple that will be used for passing the event arguments to the running
        application

        :param monitoring_context: (MonitoringApplicationContext) The monitoring context to process.

        :return: A tuple of:
                     [0] = (str) application name
                     [1] = (pd.DataFrame) current input statistics
                     [2] = (pd.DataFrame) train statistics
                     [3] = (pd.DataFrame) current input data
                     [4] = (pd.Timestamp) start time of the monitoring schedule
                     [5] = (pd.Timestamp) end time of the monitoring schedule
                     [6] = (pd.Timestamp) timestamp of the latest request
                     [7] = (str) endpoint id
                     [8] = (str) output stream uri
        """
        return (
            monitoring_context.application_name,
            cls.dict_to_histogram(monitoring_context.sample_df_stats),
            cls.dict_to_histogram(monitoring_context.feature_stats),
            monitoring_context.sample_df,
            monitoring_context.start_infer_time,
            monitoring_context.end_infer_time,
            monitoring_context.latest_request,
            monitoring_context.endpoint_id,
            monitoring_context.output_stream_uri,
        )

    @staticmethod
    def dict_to_histogram(
        histogram_dict: mlrun.common.model_monitoring.helpers.FeatureStats,
    ) -> pd.DataFrame:
        """
        Convert histogram dictionary to pandas DataFrame with feature histograms as columns

        :param histogram_dict: Histogram dictionary

        :returns: Histogram dataframe
        """

        # Create a dictionary with feature histograms as values
        histograms = {}
        for feature, stats in histogram_dict.items():
            if "hist" in stats:
                # Normalize to probability distribution of each feature
                histograms[feature] = np.array(stats["hist"][0]) / stats["count"]

        # Convert the dictionary to pandas DataFrame
        histograms = pd.DataFrame(histograms)

        return histograms
