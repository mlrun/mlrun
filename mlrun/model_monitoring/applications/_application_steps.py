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

import json
from typing import Any, Optional, Union

import mlrun.common.schemas.alert as alert_objects
import mlrun.common.schemas.model_monitoring.constants as mm_constant
import mlrun.datastore
import mlrun.model_monitoring
from mlrun.model_monitoring.helpers import get_stream_path
from mlrun.serving import GraphContext
from mlrun.serving.utils import StepToDict
from mlrun.utils import logger

from .context import MonitoringApplicationContext
from .results import ModelMonitoringApplicationMetric, ModelMonitoringApplicationResult


class _PushToMonitoringWriter(StepToDict):
    kind = "monitoring_application_stream_pusher"

    def __init__(
        self,
        project: str,
        writer_application_name: str,
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
            project=self.project, function_name=self.application_name_to_push
        )
        self.output_stream = None
        self.name = name or "PushToMonitoringWriter"

    def do(
        self,
        event: tuple[
            list[
                Union[
                    ModelMonitoringApplicationResult, ModelMonitoringApplicationMetric
                ]
            ],
            MonitoringApplicationContext,
        ],
    ) -> None:
        """
        Push application results to the monitoring writer stream.

        :param event: Monitoring result(s) to push and the original event from the controller.
        """
        self._lazy_init()
        application_results, application_context = event
        writer_event = {
            mm_constant.WriterEvent.APPLICATION_NAME: application_context.application_name,
            mm_constant.WriterEvent.ENDPOINT_ID: application_context.endpoint_id,
            mm_constant.WriterEvent.START_INFER_TIME: application_context.start_infer_time.isoformat(
                sep=" ", timespec="microseconds"
            ),
            mm_constant.WriterEvent.END_INFER_TIME: application_context.end_infer_time.isoformat(
                sep=" ", timespec="microseconds"
            ),
        }
        for result in application_results:
            data = result.to_dict()
            if isinstance(result, ModelMonitoringApplicationResult):
                writer_event[mm_constant.WriterEvent.EVENT_KIND] = (
                    mm_constant.WriterEventKind.RESULT
                )
                data[mm_constant.ResultData.CURRENT_STATS] = json.dumps(
                    application_context.sample_df_stats
                )
                writer_event[mm_constant.WriterEvent.DATA] = json.dumps(data)
            else:
                writer_event[mm_constant.WriterEvent.EVENT_KIND] = (
                    mm_constant.WriterEventKind.METRIC
                )
                writer_event[mm_constant.WriterEvent.DATA] = json.dumps(data)

            writer_event[mm_constant.WriterEvent.EVENT_KIND] = (
                mm_constant.WriterEventKind.RESULT
                if isinstance(result, ModelMonitoringApplicationResult)
                else mm_constant.WriterEventKind.METRIC
            )
            logger.info(
                f"Pushing data = {writer_event} \n to stream = {self.stream_uri}"
            )
            self.output_stream.push([writer_event])
            logger.info(f"Pushed data to {self.stream_uri} successfully")

    def _lazy_init(self):
        if self.output_stream is None:
            self.output_stream = mlrun.datastore.get_stream_pusher(
                self.stream_uri,
            )


class _PrepareMonitoringEvent(StepToDict):
    def __init__(self, context: GraphContext, application_name: str) -> None:
        """
        Class for preparing the application event for the application step.

        :param application_name: Application name.
        """
        self.graph_context = context
        self.application_name = application_name
        self.model_endpoints: dict[str, mlrun.model_monitoring.ModelEndpoint] = {}

    def do(self, event: dict[str, Any]) -> MonitoringApplicationContext:
        """
        Prepare the application event for the application step.

        :param event: Application event.
        :return: Application context.
        """
        application_context = MonitoringApplicationContext(
            graph_context=self.graph_context,
            application_name=self.application_name,
            event=event,
            model_endpoint_dict=self.model_endpoints,
        )

        self.model_endpoints.setdefault(
            application_context.endpoint_id, application_context.model_endpoint
        )

        return application_context


class _ApplicationErrorHandler(StepToDict):
    def __init__(self, project: str, name: Optional[str] = None):
        self.project = project
        self.name = name or "ApplicationErrorHandler"

    def do(self, event):
        """
        Handle model monitoring application error. This step will generate an event, describing the error.

        :param event: Application event.
        """

        error_data = {
            "Endpoint ID": event.body.endpoint_id,
            "Application Class": event.body.application_name,
            "Error": event.error,
            "Timestamp": event.timestamp,
        }
        logger.error("Error in application step", **error_data)

        event_data = alert_objects.Event(
            kind=alert_objects.EventKind.MM_APP_FAILED,
            entity=alert_objects.EventEntities(
                kind=alert_objects.EventEntityKind.MODEL_MONITORING_APPLICATION,
                project=self.project,
                ids=[f"{self.project}_{event.body.application_name}"],
            ),
            value_dict=error_data,
        )

        mlrun.get_run_db().generate_event(
            name=alert_objects.EventKind.MM_APP_FAILED, event_data=event_data
        )
        logger.info("Event generated successfully")
