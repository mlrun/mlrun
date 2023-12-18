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

    :param application_name:      (str) Name of the model monitoring application.
    :param endpoint_id:           (str) ID of the monitored model endpoint.
    :param start_infer_time:      (pd.Timestamp) Start time of the monitoring schedule.
    :param end_infer_time:        (pd.Timestamp) End time of the monitoring schedule.
    :param result_name:           (str) Name of the application result.
    :param result_value:          (float) Value of the application result.
    :param result_kind:           (ResultKindApp) Kind of application result.
    :param result_status:         (ResultStatusApp) Status of the application result.
    :param result_extra_data:     (dict) Extra data associated with the application result.
    :param _current_stats:        (dict) Current statistics of the data.
    """

    application_name: str
    endpoint_id: str
    start_infer_time: pd.Timestamp
    end_infer_time: pd.Timestamp
    result_name: str
    result_value: float
    result_kind: mm_constant.ResultKindApp
    result_status: mm_constant.ResultStatusApp
    result_extra_data: dict = dataclasses.field(default_factory=dict)
    _current_stats: dict = dataclasses.field(default_factory=dict)

    def to_dict(self):
        """
        Convert the object to a dictionary format suitable for writing.

        :returns:    (dict) Dictionary representation of the result.
        """
        return {
            mm_constant.WriterEvent.APPLICATION_NAME: self.application_name,
            mm_constant.WriterEvent.ENDPOINT_ID: self.endpoint_id,
            mm_constant.WriterEvent.START_INFER_TIME: self.start_infer_time.isoformat(
                sep=" ", timespec="microseconds"
            ),
            mm_constant.WriterEvent.END_INFER_TIME: self.end_infer_time.isoformat(
                sep=" ", timespec="microseconds"
            ),
            mm_constant.WriterEvent.RESULT_NAME: self.result_name,
            mm_constant.WriterEvent.RESULT_VALUE: self.result_value,
            mm_constant.WriterEvent.RESULT_KIND: self.result_kind,
            mm_constant.WriterEvent.RESULT_STATUS: self.result_status,
            mm_constant.WriterEvent.RESULT_EXTRA_DATA: json.dumps(
                self.result_extra_data
            ),
            mm_constant.WriterEvent.CURRENT_STATS: json.dumps(self._current_stats),
        }


class ModelMonitoringApplication(StepToDict):
    """
    Class representing a model monitoring application. Subclass this to create custom monitoring logic.

    example for very simple costume application::
        # mlrun: start-code
        class MyApp(ModelMonitoringApplication):

            def run_application(
                self,
                sample_df_stats: pd.DataFrame,
                feature_stats: pd.DataFrame,
                start_infer_time: pd.Timestamp,
                end_infer_time: pd.Timestamp,
                schedule_time: pd.Timestamp,
                latest_request: pd.Timestamp,
                endpoint_id: str,
                output_stream_uri: str,
            ) -> Union[ModelMonitoringApplicationResult, list[ModelMonitoringApplicationResult]
            ]:
                self.context.log_artifact(TableArtifact("sample_df_stats", df=sample_df_stats))
                return ModelMonitoringApplicationResult(
                    self.name,
                    endpoint_id,
                    schedule_time,
                    result_name="data_drift_test",
                    result_value=0.5,
                    result_kind=mm_constant.ResultKindApp.data_drift,
                    result_status = mm_constant.ResultStatusApp.detected,
                    result_extra_data={})

        # mlrun: end-code
    """

    kind = "monitoring_application"

    def do(self, event: dict[str, Any]) -> list[ModelMonitoringApplicationResult]:
        """
        Process the monitoring event and return application results.

        :param event:   (dict) The monitoring event to process.
        :returns:       (list[ModelMonitoringApplicationResult]) The application results.
        """
        resolved_event = self._resolve_event(event)
        if not (
            hasattr(self, "context") and isinstance(self.context, mlrun.MLClientCtx)
        ):
            self._lazy_init(app_name=resolved_event[0])
        results = self.run_application(*resolved_event)
        results = results if isinstance(results, list) else [results]
        for result in results:
            result._current_stats = event[mm_constant.ApplicationEvent.CURRENT_STATS]
        return results

    def _lazy_init(self, app_name: str):
        self.context = self._create_context_for_logging(app_name=app_name)

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

    @staticmethod
    def _resolve_event(
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
            ModelMonitoringApplication._dict_to_histogram(
                json.loads(event[mm_constant.ApplicationEvent.CURRENT_STATS])
            ),
            ModelMonitoringApplication._dict_to_histogram(
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

    def do(self, event: list[ModelMonitoringApplicationResult]) -> None:
        """
        Push application results to the monitoring writer stream.

        :param event: Monitoring result(s) to push.
        """
        self._lazy_init()
        for result in event:
            data = result.to_dict()
            logger.info(f"Pushing data = {data} \n to stream = {self.stream_uri}")
            self.output_stream.push([data])

    def _lazy_init(self):
        if self.output_stream is None:
            self.output_stream = get_stream_pusher(
                self.stream_uri,
            )
