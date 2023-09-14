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

from typing import Any, NewType

import pandas as pd
from v3io.dataplane import Client as V3IOClient
from v3io_frames.client import ClientBase as V3IOFramesClient
from v3io_frames.errors import Error as V3IOFramesError
from v3io_frames.frames_pb2 import IGNORE

import mlrun.common.model_monitoring
import mlrun.model_monitoring
import mlrun.utils.v3io_clients
from mlrun.common.schemas.model_monitoring.constants import WriterEvent
from mlrun.serving.utils import StepToDict
from mlrun.utils import logger

_TSDB_BE = "tsdb"
_TSDB_RATE = "1/s"
RawEvent = dict[str, Any]
AppResultEvent = NewType("AppResultEvent", RawEvent)


class _WriterEventError:
    pass


class _WriterEventValueError(_WriterEventError, ValueError):
    pass


class _WriterEventTypeError(_WriterEventError, TypeError):
    pass


class ModelMonitoringWriter(StepToDict):
    """
    Write monitoring app events to V3IO KV storage
    """

    kind = "monitoring_application_stream_pusher"
    _TSDB_TABLE = "app-results"

    def __init__(self, project: str) -> None:
        self.project = project
        self.name = project  # required for the deployment process
        self._v3io_container = f"users/pipelines/project/{self.name}/monitoring-apps"
        self._kv_client = self._get_v3io_client().kv
        self._tsdb_client = self._get_v3io_frames_client()
        self._create_tsdb_table()

    def _get_v3io_client(self) -> V3IOClient:
        return mlrun.utils.v3io_clients.get_v3io_client(endpoint=mlrun.mlconf.v3io_api)

    def _get_v3io_frames_client(self) -> V3IOFramesClient:
        return mlrun.utils.v3io_clients.get_frames_client(
            address=mlrun.mlconf.v3io_framesd,
            container=self._v3io_container,
        )

    def _create_tsdb_table(self) -> None:
        self._tsdb_client.create(
            backend=_TSDB_BE,
            table=self._TSDB_TABLE,
            if_exists=IGNORE,
            rate=_TSDB_RATE,
        )

    def _update_kv_db(self, event: AppResultEvent) -> None:
        event = AppResultEvent(event.copy())
        endpoint_id = event.pop(WriterEvent.ENDPOINT_ID)
        app_name = event.pop(WriterEvent.APPLICATION_NAME)
        self._kv_client.put(
            container=self._v3io_container,
            table_path=endpoint_id,
            key=app_name,
            attributes=event,
        )
        logger.info("Updated V3IO KV successfully", key=app_name)

    def _update_tsdb(self, event: AppResultEvent) -> None:
        try:
            self._tsdb_client.write(
                backend=_TSDB_BE,
                table=self._TSDB_TABLE,
                dfs=pd.DataFrame.from_records([event]),
                index_cols=[
                    WriterEvent.SCHEDULE_TIME,
                    WriterEvent.ENDPOINT_ID,
                    WriterEvent.APPLICATION_NAME,
                ],
            )
            logger.info("Updated V3IO TSDB successfully", table=self._TSDB_TABLE)
        except V3IOFramesError as err:
            logger.warn(
                "Could not write drift measures to TSDB",
                err=err,
                table=self._TSDB_TABLE,
                endpoint=event[WriterEvent.ENDPOINT_ID],
            )

    @staticmethod
    def _reconstruct_event(event: RawEvent) -> AppResultEvent:
        try:
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
        except KeyError as err:
            raise _WriterEventValueError(
                "The received event misses some keys compared to the expected "
                "monitoring application event schema"
            ) from err
        except TypeError as err:
            raise _WriterEventTypeError(
                f"The event is of type: {type(event)}, expected a dictionary"
            ) from err

    def do(self, event: RawEvent) -> None:
        event = self._reconstruct_event(event)
        logger.info("Starting to write event", event=event)
        self._update_tsdb(event)
        self._update_kv_db(event)
        logger.info("Completed event DB writes")
