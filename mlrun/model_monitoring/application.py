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

import dataclasses
import json
import re
from abc import ABC, abstractmethod
from typing import Union, cast

import numpy as np
import pandas as pd

import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring.constants as mm_constant
import mlrun.utils.v3io_clients
from mlrun.model_monitoring.applications.context import MonitoringApplicationContext
from mlrun.serving.utils import StepToDict


@dataclasses.dataclass
class ModelMonitoringApplicationResult:
    """
    Class representing the result of a custom model monitoring application.

    :param name:           (str) Name of the application result. This name must be
                            unique for each metric in a single application
                            (name must be of the format [a-zA-Z_][a-zA-Z0-9_]*).
    :param value:          (float) Value of the application result.
    :param kind:           (ResultKindApp) Kind of application result.
    :param status:         (ResultStatusApp) Status of the application result.
    :param extra_data:     (dict) Extra data associated with the application result.
    """

    name: str
    value: float
    kind: mm_constant.ResultKindApp
    status: mm_constant.ResultStatusApp
    extra_data: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        pat = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
        if not re.fullmatch(pat, self.name):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Attribute name must be of the format [a-zA-Z_][a-zA-Z0-9_]*"
            )

    def to_dict(self):
        """
        Convert the object to a dictionary format suitable for writing.

        :returns:    (dict) Dictionary representation of the result.
        """
        return {
            mm_constant.WriterEvent.RESULT_NAME: self.name,
            mm_constant.WriterEvent.RESULT_VALUE: self.value,
            mm_constant.WriterEvent.RESULT_KIND: self.kind,
            mm_constant.WriterEvent.RESULT_STATUS: self.status,
            mm_constant.WriterEvent.RESULT_EXTRA_DATA: json.dumps(self.extra_data),
        }


class ModelMonitoringApplicationBase(StepToDict, ABC):
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


        # mlrun: end-code
    """

    kind = "monitoring_application"

    def do(
        self, event: MonitoringApplicationContext
    ) -> tuple[list[ModelMonitoringApplicationResult], MonitoringApplicationContext]:
        """
        Process the monitoring event and return application results.

        :param event:   (dict) The monitoring event to process.
        :returns:       (list[ModelMonitoringApplicationResult], dict) The application results
                        and the original event for the application.
        """
        resolved_event = self._resolve_event(event)
        if not (
            hasattr(self, "context") and isinstance(self.context, mlrun.MLClientCtx)
        ):
            self._lazy_init(event)
        results = self.do_tracking(*resolved_event)
        results = results if isinstance(results, list) else [results]
        return results, event

    def _lazy_init(self, monitoring_context: MonitoringApplicationContext):
        self.context = cast(mlrun.MLClientCtx, monitoring_context)

    @abstractmethod
    def do_tracking(
        self,
        application_name: str,
        sample_df_stats: mlrun.common.model_monitoring.helpers.FeatureStats,
        feature_stats: mlrun.common.model_monitoring.helpers.FeatureStats,
        sample_df: pd.DataFrame,
        start_infer_time: pd.Timestamp,
        end_infer_time: pd.Timestamp,
        latest_request: pd.Timestamp,
        endpoint_id: str,
        output_stream_uri: str,
    ) -> Union[
        ModelMonitoringApplicationResult, list[ModelMonitoringApplicationResult]
    ]:
        """
        Implement this method with your custom monitoring logic.

        :param application_name:        (str) the app name
        :param sample_df_stats:         (FeatureStats) The new sample distribution dictionary.
        :param feature_stats:           (FeatureStats) The train sample distribution dictionary.
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
        monitoring_context: MonitoringApplicationContext,
    ) -> tuple[
        str,
        mlrun.common.model_monitoring.helpers.FeatureStats,
        mlrun.common.model_monitoring.helpers.FeatureStats,
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

        :param event: dictionary with all the incoming data

        :return: A tuple of:
                     [0] = (str) application name
                     [1] = (dict) current input statistics
                     [2] = (dict) train statistics
                     [3] = (pd.DataFrame) current input data
                     [4] = (pd.Timestamp) start time of the monitoring schedule
                     [5] = (pd.Timestamp) end time of the monitoring schedule
                     [6] = (pd.Timestamp) timestamp of the latest request
                     [7] = (str) endpoint id
                     [8] = (str) output stream uri
        """
        return (
            monitoring_context.application_name,
            monitoring_context.sample_df_stats,
            monitoring_context.feature_stats,
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
