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
from typing import Optional

import pandas as pd

import mlrun.common.helpers
import mlrun.common.model_monitoring.helpers
import mlrun.common.schemas.model_monitoring.constants as mm_constant
import mlrun.utils.v3io_clients
from mlrun.datastore import get_stream_pusher
from mlrun.datastore.targets import ParquetTarget
from mlrun.model_monitoring.helpers import get_stream_path
from mlrun.serving.utils import StepToDict
from mlrun.utils import logger

from ..application import ModelMonitoringApplicationResult


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

    def do(self, event: tuple[list[ModelMonitoringApplicationResult], dict]) -> None:
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


@dataclasses.dataclass
class MonitoringApplicationContext(mlrun.MLClientCtx):
    """
    Application context object.

    """

    application_name: str = None
    sample_df_stats: mlrun.common.model_monitoring.helpers.FeatureStats = None
    feature_stats: mlrun.common.model_monitoring.helpers.FeatureStats = None
    sample_df: pd.DataFrame = None
    start_infer_time: pd.Timestamp = None
    end_infer_time: pd.Timestamp = None
    latest_request: pd.Timestamp = None
    endpoint_id: str = None
    output_stream_uri: str = None
    data: dict = None  # for inputs, outputs, and other data
    autocommit = None
    tmp = None
    log_stream = None

    def __post_init__(self):
        pat = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
        if not re.fullmatch(pat, self.application_name):
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Attribute name must be of the format [a-zA-Z_][a-zA-Z0-9_]*"
            )

    def from_dict(
        cls,
        attrs: dict,
        rundb="",
        autocommit=False,
        tmp="",
        host=None,
        log_stream=None,
        is_api=False,
        store_run=True,
        include_status=False,
        context=None,
    ) -> "MonitoringApplicationContext":
        """
        Converting the event into a single tuple that will be used for passing the event arguments to the running
        application

        :param event: dictionary with all the incoming data

        """

        # Call the parent class from_dict method - maybe we should take only the relevant attributes, and not all.
        if not context:
            self: MonitoringApplicationContext = super(cls).from_dict(
                attrs,
                rundb,
                autocommit,
                tmp,
                host,
                log_stream,
                is_api,
                store_run,
                include_status,
            )
        else:
            self: MonitoringApplicationContext = context

        self.start_infer_time = pd.Timestamp(
            attrs.get(mm_constant.ApplicationEvent.START_INFER_TIME)
        )
        self.end_infer_time = pd.Timestamp(
            attrs[mm_constant.ApplicationEvent.END_INFER_TIME]
        )
        self.application_name = attrs[mm_constant.ApplicationEvent.APPLICATION_NAME]
        self.sample_df_stats = json.loads(
            attrs[mm_constant.ApplicationEvent.CURRENT_STATS]
        )
        self.feature_stats = json.loads(
            attrs[mm_constant.ApplicationEvent.FEATURE_STATS]
        )
        self.sample_df = ParquetTarget(
            path=attrs[mm_constant.ApplicationEvent.SAMPLE_PARQUET_PATH]
        ).as_df(
            start_time=self.start_infer_time,
            end_time=self.end_infer_time,
            time_column="timestamp",
        )
        self.latest_request = pd.Timestamp(
            attrs[mm_constant.ApplicationEvent.LAST_REQUEST]
        )
        self.endpoint_id = attrs[mm_constant.ApplicationEvent.ENDPOINT_ID]
        self.output_stream_uri = attrs[mm_constant.ApplicationEvent.OUTPUT_STREAM_URI]
        self.data = {}

        return self

    def __getitem__(self, key):
        return self.data.get(key)

    def __setitem__(self, key, value):
        self.data[key] = value

    def to_dict(self):
        """TODO: edit"""
        a = super(self).to_dict()
        return a


class _PrepareToApplication:
    def __init__(self, application_name: str):
        """
        Class for preparing the application event for the application step.

        :param application_name: Application name.
        """

        self.context = self._create_mlrun_context(application_name)
        # create context for the application

    def do(self, event: dict[str, dict]) -> MonitoringApplicationContext:
        """
        Prepare the application event for the application step.

        :param event: Application event.
        :return: Application event.
        """
        if not hasattr(event, "metadata"):
            application_context = MonitoringApplicationContext().from_dict(
                event, context=self.context
            )
        else:
            application_context = MonitoringApplicationContext().from_dict(event)
        return application_context

    @staticmethod
    def _create_mlrun_context(app_name: str):
        context = mlrun.get_or_create_ctx(
            f"{app_name}-logger",
            upload_artifacts=True,
            labels={"workflow": "model-monitoring-app-logger"},
        )
        return context
