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

import json
from http import HTTPStatus

from v3io.dataplane import Client as V3IOClient

import mlrun.common.model_monitoring
import mlrun.model_monitoring
import mlrun.model_monitoring.stores.tsdb.v3io.v3io_tsdb
import mlrun.utils.v3io_clients
from mlrun.common.schemas.model_monitoring.constants import (
    AppResultEvent,
    RawEvent,
    ResultStatusApp,
    WriterEvent,
)
from mlrun.common.schemas.notification import NotificationKind, NotificationSeverity
from mlrun.serving.utils import StepToDict
from mlrun.utils import logger
from mlrun.utils.notifications.notification_pusher import CustomNotificationPusher

_TSDB_TABLE = "app-results"


class _WriterEventError:
    pass


class _WriterEventValueError(_WriterEventError, ValueError):
    pass


class _WriterEventTypeError(_WriterEventError, TypeError):
    pass


class _Notifier:
    def __init__(
        self,
        event: AppResultEvent,
        notification_pusher: CustomNotificationPusher,
        severity: NotificationSeverity = NotificationSeverity.WARNING,
    ) -> None:
        """
        Event notifier - send push notification when appropriate to the notifiers in
        `notification pusher`.
        Note that if you use a Slack App webhook, you need to define it as an MLRun secret
        `SLACK_WEBHOOK`.
        """
        self._event = event
        self._custom_notifier = notification_pusher
        self._severity = severity

    def _should_send_event(self) -> bool:
        return self._event[WriterEvent.RESULT_STATUS] >= ResultStatusApp.detected

    def _generate_message(self) -> str:
        return f"""\
The monitoring app `{self._event[WriterEvent.APPLICATION_NAME]}` \
of kind `{self._event[WriterEvent.RESULT_KIND]}` \
detected a problem in model endpoint ID `{self._event[WriterEvent.ENDPOINT_ID]}` \
at time `{self._event[WriterEvent.START_INFER_TIME]}`.

Result data:
Name: `{self._event[WriterEvent.RESULT_NAME]}`
Value: `{self._event[WriterEvent.RESULT_VALUE]}`
Status: `{self._event[WriterEvent.RESULT_STATUS]}`
Extra data: `{self._event[WriterEvent.RESULT_EXTRA_DATA]}`\
"""

    def notify(self) -> None:
        """Send notification if appropriate"""
        if not self._should_send_event():
            logger.debug("Not sending a notification")
            return
        message = self._generate_message()
        self._custom_notifier.push(message=message, severity=self._severity)
        logger.debug("A notification should have been sent")


class ModelMonitoringWriter(StepToDict):
    """
    Write monitoring app events to V3IO KV storage
    """

    kind = "monitoring_application_stream_pusher"

    def __init__(self, project: str) -> None:
        self.project = project
        self.name = project  # required for the deployment process
        self._v3io_container = self.get_v3io_container(self.name)
        self._kv_client = self._get_v3io_client().kv
        # self._tsdb_client = self._get_v3io_frames_client(self._v3io_container)
        self._custom_notifier = CustomNotificationPusher(
            notification_types=[NotificationKind.slack]
        )
        # self._create_tsdb_table()
        self._kv_schemas = []

    @staticmethod
    def get_v3io_container(project_name: str) -> str:
        return f"users/pipelines/{project_name}/monitoring-apps"

    @staticmethod
    def _get_v3io_client() -> V3IOClient:
        return mlrun.utils.v3io_clients.get_v3io_client(
            endpoint=mlrun.mlconf.v3io_api,
        )

    def _update_kv_db(self, event: AppResultEvent) -> None:
        event = AppResultEvent(event.copy())
        endpoint_id = event.pop(WriterEvent.ENDPOINT_ID)
        app_name = event.pop(WriterEvent.APPLICATION_NAME)
        metric_name = event.pop(WriterEvent.RESULT_NAME)
        attributes = {metric_name: json.dumps(event)}
        self._kv_client.update(
            container=self._v3io_container,
            table_path=endpoint_id,
            key=app_name,
            attributes=attributes,
        )
        if endpoint_id not in self._kv_schemas:
            self._generate_kv_schema(endpoint_id)
        logger.info("Updated V3IO KV successfully", key=app_name)

    def _generate_kv_schema(self, endpoint_id: str):
        """Generate V3IO KV schema file which will be used by the model monitoring applications dashboard in Grafana."""
        fields = [
            {"name": WriterEvent.RESULT_NAME, "type": "string", "nullable": False}
        ]
        res = self._kv_client.create_schema(
            container=self._v3io_container,
            table_path=endpoint_id,
            key=WriterEvent.APPLICATION_NAME,
            fields=fields,
        )
        if res.status_code != HTTPStatus.OK.value:
            raise mlrun.errors.MLRunBadRequestError(
                f"Couldn't infer schema for endpoint {endpoint_id} which is required for Grafana dashboards"
            )
        else:
            logger.info(
                "Generated V3IO KV schema successfully", endpoint_id=endpoint_id
            )
            self._kv_schemas.append(endpoint_id)

    def _update_tsdb(self, event: AppResultEvent) -> None:
        tsdb_store = mlrun.model_monitoring.get_tsdb_store(
            project=self.project,
            table=_TSDB_TABLE,
            container=self._v3io_container,
            create_table=True,
        )

        tsdb_store.write_application_event(event=event)

    @staticmethod
    def _reconstruct_event(event: RawEvent) -> AppResultEvent:
        """
        Modify the raw event into the expected monitoring application event
        schema as defined in `mlrun.common.schemas.model_monitoring.constants.WriterEvent`
        """
        try:
            result_event = AppResultEvent(
                {key: event[key] for key in WriterEvent.list()}
            )
            result_event[WriterEvent.CURRENT_STATS] = json.loads(
                event[WriterEvent.CURRENT_STATS]
            )
            return result_event
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
        _Notifier(event=event, notification_pusher=self._custom_notifier).notify()
        logger.info("Completed event DB writes")
