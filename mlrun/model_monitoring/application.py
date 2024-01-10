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
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd

import mlrun.common.helpers
import mlrun.common.schemas.model_monitoring
import mlrun.common.schemas.model_monitoring.constants as mm_constant
import mlrun.utils.v3io_clients
from mlrun.datastore import get_stream_pusher
from mlrun.datastore.targets import ParquetTarget
from mlrun.model_monitoring.helpers import get_stream_path
from mlrun.serving.utils import StepToDict
from mlrun.utils import logger


@dataclasses.dataclass
class ModelMonitoringApplicationResult:
    """
    Class representing the result of a custom model monitoring application.

    :param name:           (str) Name of the application result. This name must be
                            unique for each metric in a single application.
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
                sample_df_stats: pd.DataFrame,
                feature_stats: pd.DataFrame,
                start_infer_time: pd.Timestamp,
                end_infer_time: pd.Timestamp,
                schedule_time: pd.Timestamp,
                latest_request: pd.Timestamp,
                endpoint_id: str,
                output_stream_uri: str,
            ) -> ModelMonitoringApplicationResult:
                self.context.log_artifact(TableArtifact("sample_df_stats", df=sample_df_stats))
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
        self, event: dict[str, Any]
    ) -> Tuple[list[ModelMonitoringApplicationResult], dict]:
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
            self._lazy_init(app_name=resolved_event[0])
        results = self.do_tracking(*resolved_event)
        results = results if isinstance(results, list) else [results]
        return results, event

    def _lazy_init(self, app_name: str):
        self.context = self._create_context_for_logging(app_name=app_name)

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
        ModelMonitoringApplicationResult, list[ModelMonitoringApplicationResult]
    ]:
        """
        Implement this method with your custom monitoring logic.

        :param application_name:         (str) the app name
        :param sample_df_stats:         (pd.DataFrame) The new sample distribution DataFrame.
        :param feature_stats:           (pd.DataFrame) The train sample distribution DataFrame.
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
        event: dict[str, Any],
    ) -> Tuple[
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

        :param event: dictionary with all the incoming data

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
        start_time = pd.Timestamp(event[mm_constant.ApplicationEvent.START_INFER_TIME])
        end_time = pd.Timestamp(event[mm_constant.ApplicationEvent.END_INFER_TIME])
        return (
            event[mm_constant.ApplicationEvent.APPLICATION_NAME],
            cls._dict_to_histogram(
                json.loads(event[mm_constant.ApplicationEvent.CURRENT_STATS])
            ),
            cls._dict_to_histogram(
                json.loads(event[mm_constant.ApplicationEvent.FEATURE_STATS])
            ),
            ParquetTarget(
                path=event[mm_constant.ApplicationEvent.SAMPLE_PARQUET_PATH]
            ).as_df(start_time=start_time, end_time=end_time, time_column="timestamp"),
            start_time,
            end_time,
            pd.Timestamp(event[mm_constant.ApplicationEvent.LAST_REQUEST]),
            event[mm_constant.ApplicationEvent.ENDPOINT_ID],
            event[mm_constant.ApplicationEvent.OUTPUT_STREAM_URI],
        )

    @staticmethod
    def _create_context_for_logging(app_name: str):
        context = mlrun.get_or_create_ctx(
            f"{app_name}-logger",
            upload_artifacts=True,
            labels={"workflow": "model-monitoring-app-logger"},
        )
        return context

    @staticmethod
    def _dict_to_histogram(histogram_dict: dict[str, dict[str, Any]]) -> pd.DataFrame:
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


class PushToMonitoringWriter(StepToDict):
    kind = "monitoring_application_stream_pusher"

    def __init__(
        self,
        project: Optional[str] = None,
        writer_application_name: Optional[str] = None,
        stream_uri: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Class for pushing application results to the monitoring writer stream.

        :param project:                     Project name.
        :param writer_application_name:     Writer application name.
        :param stream_uri:                  Stream URI for pushing results.
        :param name:                        Name of the PushToMonitoringWriter
                                            instance default to PushToMonitoringWriter.
        """
        self.project = project
        self.application_name_to_push = writer_application_name
        self.stream_uri = stream_uri or get_stream_path(
            project=self.project, application_name=self.application_name_to_push
        )
        self.output_stream = None
        self.name = name or "PushToMonitoringWriter"

    def do(self, event: Tuple[list[ModelMonitoringApplicationResult], dict]) -> None:
        """
        Push application results to the monitoring writer stream.

        :param event: Monitoring result(s) to push and the original event from the controller.
        """
        self._lazy_init()
        application_results, application_event = event
        metadata = {
            mm_constant.WriterEvent.APPLICATION_NAME: application_event[
                mm_constant.ApplicationEvent.APPLICATION_NAME
            ],
            mm_constant.WriterEvent.ENDPOINT_ID: application_event[
                mm_constant.ApplicationEvent.ENDPOINT_ID
            ],
            mm_constant.WriterEvent.START_INFER_TIME: application_event[
                mm_constant.ApplicationEvent.START_INFER_TIME
            ],
            mm_constant.WriterEvent.END_INFER_TIME: application_event[
                mm_constant.ApplicationEvent.END_INFER_TIME
            ],
            mm_constant.WriterEvent.CURRENT_STATS: json.dumps(
                application_event[mm_constant.ApplicationEvent.CURRENT_STATS]
            ),
        }
        for result in application_results:
            data = result.to_dict()
            data.update(metadata)
            logger.info(f"Pushing data = {data} \n to stream = {self.stream_uri}")
            self.output_stream.push([data])

    def _lazy_init(self):
        if self.output_stream is None:
            self.output_stream = get_stream_pusher(
                self.stream_uri,
            )
