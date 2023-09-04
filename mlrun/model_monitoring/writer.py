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

import os
from typing import Any, NewType, Tuple

import pandas as pd
from v3io_frames.client import ClientBase
from v3io_frames.errors import Error as V3IOFramesError

import mlrun.common.model_monitoring
import mlrun.model_monitoring
import mlrun.utils.v3io_clients
from mlrun.common.schemas.model_monitoring import FileTargetKind
from mlrun.common.schemas.model_monitoring.constants import WriterEvent
from mlrun.serving.utils import StepToDict
from mlrun.utils import logger

_V3IO_ACCESS_KEY_NAME = "V3IO_ACCESS_KEY"

RawEvent = dict[str, Any]
AppResultEvent = NewType("AppResultEvent", RawEvent)


class ModelMonitoringWriter(StepToDict):
    """
    Write monitoring app events to V3IO KV storage
    """

    kind = "monitoring_application_stream_pusher"

    def __init__(self, project: str) -> None:
        self.project = project
        self._kv_db = self._get_kv_db()

        self._v3io_access_key = os.getenv(_V3IO_ACCESS_KEY_NAME)
        self._tsdb_path, self._frames_client = self._get_v3io_frames_client()

    def _get_kv_db(self) -> mlrun.model_monitoring.stores.ModelEndpointStore:
        return mlrun.model_monitoring.get_model_endpoint_store(project=self.project)

    def _get_v3io_frames_client(self) -> Tuple[str, ClientBase]:
        tsdb_path = mlrun.mlconf.get_model_monitoring_file_target_path(
            project=self.project,
            kind=FileTargetKind.EVENTS,
        )
        (
            _,
            tsdb_container,
            _,
        ) = mlrun.common.model_monitoring.helpers.parse_model_endpoint_store_prefix(
            tsdb_path
        )
        return tsdb_path, mlrun.utils.v3io_clients.get_frames_client(
            address=mlrun.mlconf.v3io_framesd,
            container=tsdb_container,
            token=self._v3io_access_key,
        )

    def _update_kv_db(self, event: AppResultEvent) -> None:
        event = AppResultEvent(event.copy())
        endpoint_id = event.pop(WriterEvent.ENDPOINT_ID)
        self._kv_db.update_model_endpoint(endpoint_id=endpoint_id, attributes=event)

    def _update_tsdb(self, event: AppResultEvent) -> None:
        try:
            self._frames_client.write(
                backend="tsdb",
                table=self._tsdb_path,
                dfs=pd.DataFrame.from_records([event]),
                index_cols=[
                    WriterEvent.SCHEDULE_TIME,
                    WriterEvent.ENDPOINT_ID,
                    WriterEvent.APPLICATION_NAME,
                ],
            )
        except V3IOFramesError as err:
            logger.warn(
                "Could not write drift measures to TSDB",
                err=err,
                tsdb_path=self._tsdb_path,
                endpoint=event[WriterEvent.ENDPOINT_ID],
            )

    @staticmethod
    def _reconstruct_event(event: RawEvent) -> AppResultEvent:
        return AppResultEvent(
            {
                key: event[key]
                for key in (
                    WriterEvent.ENDPOINT_ID,
                    WriterEvent.SCHEDULE_TIME,
                    WriterEvent.APPLICATION_NAME,
                    WriterEvent.RESULT_NAME,
                    WriterEvent.RESULT_KIND,
                    WriterEvent.RESULT_VALUE,
                    WriterEvent.RESULT_STATUS,
                    WriterEvent.RESULT_EXTRA_DATA,
                )
            }
        )

    def do(self, event: RawEvent) -> None:
        event = self._reconstruct_event(event)
        self._update_tsdb(event)
        self._update_kv_db(event)
