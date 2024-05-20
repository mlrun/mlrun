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
import typing
from typing import Optional

import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring.constants as mm_constant
import mlrun.datastore
import mlrun.utils.v3io_clients
from mlrun.model_monitoring.helpers import get_stream_path
from mlrun.serving.utils import StepToDict
from mlrun.utils import logger

from .context import MonitoringApplicationContext
from .results import ModelMonitoringApplicationMetric, ModelMonitoringApplicationResult


class _PushToMonitoringWriter(StepToDict):
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
            project=self.project, function_name=self.application_name_to_push
        )
        self.output_stream = None
        self.name = name or "PushToMonitoringWriter"

    def do(
        self,
        event: tuple[
            list[
                typing.Union[
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

    def _lazy_init(self):
        if self.output_stream is None:
            self.output_stream = mlrun.datastore.get_stream_pusher(
                self.stream_uri,
            )


class _PrepareMonitoringEvent(StepToDict):
    def __init__(self, application_name: str):
        """
        Class for preparing the application event for the application step.

        :param application_name: Application name.
        """

        self.context = self._create_mlrun_context(application_name)
        self.model_endpoints = {}

    def do(self, event: dict[str, dict]) -> MonitoringApplicationContext:
        """
        Prepare the application event for the application step.

        :param event: Application event.
        :return: Application event.
        """
        if not event.get("mlrun_context"):
            application_context = MonitoringApplicationContext().from_dict(
                event,
                context=self.context,
                model_endpoint_dict=self.model_endpoints,
            )
        else:
            application_context = MonitoringApplicationContext().from_dict(event)
        self.model_endpoints.setdefault(
            application_context.endpoint_id, application_context.model_endpoint
        )
        return application_context

    @staticmethod
    def _create_mlrun_context(app_name: str):
        context = mlrun.get_or_create_ctx(
            f"{app_name}-logger",
            upload_artifacts=True,
        )
        context.__class__ = MonitoringApplicationContext
        return context
