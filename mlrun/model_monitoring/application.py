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
#

import dataclasses
import json
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

import mlrun.common.helpers
import mlrun.common.schemas.model_monitoring
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

    :param application_name:     (str) Name of the model monitoring application.
    :param endpoint_id:          (str) ID of the monitored model endpoint.
    :param schedule_time:        (pd.Timestamp)Timestamp of the monitoring schedule.
    :param result_name:          (str) Name of the application result.
    :param result_value:         (float) Value of the application result.
    :param result_kind:          (ResultKindApp) Kind of application result.
    :param result_status:        (ResultStatusApp) Status of the application result.
    :param result_extra_data:    (dict) Extra data associated with the application result.

    """

    application_name: str
    endpoint_id: str
    schedule_time: pd.Timestamp
    result_name: str
    result_value: float
    result_kind: mlrun.common.schemas.model_monitoring.constants.ResultKindApp
    result_status: mlrun.common.schemas.model_monitoring.constants.ResultStatusApp
    result_extra_data: dict

    def to_dict(self):
        """
        Convert the object to a dictionary format suitable for writing.

        :returns:    (dict) Dictionary representation of the result.
        """
        return {
            mlrun.common.schemas.model_monitoring.constants.WriterEvent.APPLICATION_NAME: self.application_name,
            mlrun.common.schemas.model_monitoring.constants.WriterEvent.ENDPOINT_ID: self.endpoint_id,
            mlrun.common.schemas.model_monitoring.constants.WriterEvent.SCHEDULE_TIME: self.schedule_time.isoformat(
                sep=" ", timespec="microseconds"
            ),
            mlrun.common.schemas.model_monitoring.constants.WriterEvent.RESULT_NAME: self.result_name,
            mlrun.common.schemas.model_monitoring.constants.WriterEvent.RESULT_VALUE: self.result_value,
            mlrun.common.schemas.model_monitoring.constants.WriterEvent.RESULT_KIND: self.result_kind.value,
            mlrun.common.schemas.model_monitoring.constants.WriterEvent.RESULT_STATUS: self.result_status.value,
            mlrun.common.schemas.model_monitoring.constants.WriterEvent.RESULT_EXTRA_DATA: json.dumps(
                self.result_extra_data
            ),
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
                sample_df: pd.DataFrame,
                schedule_time: pd.Timestamp,
                latest_request: pd.Timestamp,
                endpoint_id: str,
                output_stream_uri: str,
            ) -> typing.Union[ModelMonitoringApplicationResult, typing.List[ModelMonitoringApplicationResult]
            ]:
                self.context.log_artifact(TableArtifact("sample_df_stats", df=sample_df_stats))
                return ModelMonitoringApplicationResult(
                    self.name,
                    endpoint_id,
                    schedule_time,
                    result_name="data_drift_test",
                    result_value=0.5,
                    result_kind=mlrun.common.schemas.model_monitoring.constants.ResultKindApp.data_drift,
                    result_status = mlrun.common.schemas.model_monitoring.constants.ResultStatusApp.detected,
                    result_extra_data={})

        # mlrun: end-code
    """

    kind = "monitoring_application"

    def do(self, event: Dict[str, Any]):
        """
        Process the monitoring event and return application results.

        :param event:   (dict) The monitoring event to process.
        :returns:       (List[ModelMonitoringApplicationResult]) The application results.
        """
        resolved_event = self._resolve_event(event)
        if not (
            hasattr(self, "context") and isinstance(self.context, mlrun.MLClientCtx)
        ):
            self._lazy_init(app_name=resolved_event[0])
        return self.run_application(*resolved_event)

    def _lazy_init(self, app_name: str):
        self.context = self._create_context_for_logging(app_name=app_name)

    def run_application(
        self,
        application_name: str,
        sample_df_stats: pd.DataFrame,
        feature_stats: pd.DataFrame,
        sample_df: pd.DataFrame,
        schedule_time: pd.Timestamp,
        latest_request: pd.Timestamp,
        endpoint_id: str,
        output_stream_uri: str,
    ) -> Union[
        ModelMonitoringApplicationResult, List[ModelMonitoringApplicationResult]
    ]:
        """
        Implement this method with your custom monitoring logic.

        :param application_name     (str) the app name
        :param sample_df_stats:     (pd.DataFrame) The new sample distribution DataFrame.
        :param feature_stats:       (pd.DataFrame) The train sample distribution DataFrame.
        :param sample_df:           (pd.DataFrame) The new sample DataFrame.
        :param schedule_time:       (pd.Timestamp) Timestamp of the monitoring schedule.
        :param latest_request:      (pd.Timestamp) Timestamp of the latest request on this endpoint_id.
        :param endpoint_id:         (str) ID of the monitored model endpoint
        :param output_stream_uri:   (str) URI of the output stream for results

        :returns:                   (ModelMonitoringApplicationResult) or
                                    (List[ModelMonitoringApplicationResult]) of the application results.
        """
        raise NotImplementedError

    @staticmethod
    def _resolve_event(
        event: Dict[str, Any],
    ) -> Tuple[
        str,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Timestamp,
        pd.Timestamp,
        str,
        str,
    ]:
        """
        Converting the event into a single tuple that will be be used for passing the event arguments to the running
        application

        :param event: dictionary with all the incoming data

        :return: A tuple of:
                     [0] = (str) application name
                     [1] = (pd.DataFrame) current input statistics
                     [2] = (pd.DataFrame) train statistics
                     [3] = (pd.DataFrame) current input data
                     [4] = (pd.Timestamp) timestamp of batch schedule time
                     [5] = (pd.Timestamp) timestamp of the latest request
                     [6] = (str) endpoint id
                     [7] = (str) output stream uri
        """
        return (
            event[
                mlrun.common.schemas.model_monitoring.constants.ApplicationEvent.APPLICATION_NAME
            ],
            ModelMonitoringApplication._dict_to_histogram(
                json.loads(
                    event[
                        mlrun.common.schemas.model_monitoring.constants.ApplicationEvent.CURRENT_STATS
                    ]
                )
            ),
            ModelMonitoringApplication._dict_to_histogram(
                json.loads(
                    event[
                        mlrun.common.schemas.model_monitoring.constants.ApplicationEvent.FEATURE_STATS
                    ]
                )
            ),
            ParquetTarget(
                path=event[
                    mlrun.common.schemas.model_monitoring.constants.ApplicationEvent.SAMPLE_PARQUET_PATH
                ]
            ).as_df(),
            pd.Timestamp(
                event[
                    mlrun.common.schemas.model_monitoring.constants.ApplicationEvent.SCHEDULE_TIME
                ]
            ),
            pd.Timestamp(
                event[
                    mlrun.common.schemas.model_monitoring.constants.ApplicationEvent.LAST_REQUEST
                ]
            ),
            event[
                mlrun.common.schemas.model_monitoring.constants.ApplicationEvent.ENDPOINT_ID
            ],
            event[
                mlrun.common.schemas.model_monitoring.constants.ApplicationEvent.OUTPUT_STREAM_URI
            ],
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
    def _dict_to_histogram(histogram_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
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
        project: str = None,
        writer_application_name: str = None,
        stream_uri: str = None,
        name: str = None,
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

    def do(
        self,
        event: Union[
            ModelMonitoringApplicationResult, List[ModelMonitoringApplicationResult]
        ],
    ):
        """
        Push application results to the monitoring writer stream.

        :param event: Monitoring result(s) to push.
        """
        self._lazy_init()
        event = event if isinstance(event, List) else [event]
        for result in event:
            data = result.to_dict()
            logger.info(f"Pushing data = {data} \n to stream = {self.stream_uri}")
            self.output_stream.push([data])

    def _lazy_init(self):
        if self.output_stream is None:
            self.output_stream = get_stream_pusher(
                self.stream_uri,
            )
