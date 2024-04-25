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
from typing import Any, NewType

import mlrun.common.model_monitoring
import mlrun.common.schemas.alert as alert_constants
import mlrun.model_monitoring
import mlrun.model_monitoring.db.stores
from mlrun.common.schemas.model_monitoring.constants import ResultStatusApp, WriterEvent
from mlrun.common.schemas.notification import NotificationKind, NotificationSeverity
from mlrun.serving.utils import StepToDict
from mlrun.utils import logger
from mlrun.utils.notifications.notification_pusher import CustomNotificationPusher

_TSDB_BE = "tsdb"
_TSDB_RATE = "1/s"
_TSDB_TABLE = "app-results"
_RawEvent = dict[str, Any]
_AppResultEvent = NewType("_AppResultEvent", _RawEvent)


class _WriterEventError:
    pass


class _WriterEventValueError(_WriterEventError, ValueError):
    pass


class _WriterEventTypeError(_WriterEventError, TypeError):
    pass


class _Notifier:
    def __init__(
        self,
        event: _AppResultEvent,
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

        self._custom_notifier = CustomNotificationPusher(
            notification_types=[NotificationKind.slack]
        )

    def _update_kv_db(self, event: _AppResultEvent) -> None:
        event = _AppResultEvent(event.copy())
        application_result_store = mlrun.model_monitoring.get_store_object(
            project=self.project
        )
        application_result_store.write_application_result(event=event)

    def _update_tsdb(self, event: _AppResultEvent) -> None:
        event = _AppResultEvent(event.copy())
        # TODO: remove create_table=True in 1.9.0 (backwards compatibility)
        tsdb_store = mlrun.model_monitoring.get_tsdb_target(
            project=self.project,
            create_table=True,
        )

        tsdb_store.write_application_event(event=event)

    @staticmethod
    def _generate_event_on_drift(
        uid: str, drift_status: str, drift_value: float, project_name: str
    ):
        if (
            drift_status == ResultStatusApp.detected
            or drift_status == ResultStatusApp.potential_detection
        ):
            entity = {
                "kind": alert_constants.EventEntityKind.MODEL,
                "project": project_name,
                "id": uid,
            }
            event_kind = (
                alert_constants.EventKind.DRIFT_DETECTED
                if drift_status == ResultStatusApp.detected
                else alert_constants.EventKind.DRIFT_SUSPECTED
            )
            event_data = mlrun.common.schemas.Event(
                kind=event_kind, entity=entity, value=drift_value
            )
            mlrun.get_run_db().generate_event(event_kind, event_data)

    @staticmethod
    def _reconstruct_event(event: _RawEvent) -> _AppResultEvent:
        """
        Modify the raw event into the expected monitoring application event
        schema as defined in `mlrun.common.schemas.model_monitoring.constants.WriterEvent`
        """
        try:
            result_event = _AppResultEvent(
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

    def do(self, event: _RawEvent) -> None:
        event = self._reconstruct_event(event)
        logger.info("Starting to write event", event=event)
        self._update_tsdb(event)
        self._update_kv_db(event)
        _Notifier(event=event, notification_pusher=self._custom_notifier).notify()

        if mlrun.mlconf.alerts.mode == mlrun.common.schemas.alert.AlertsModes.enabled:
            self._generate_event_on_drift(
                event[WriterEvent.ENDPOINT_ID],
                event[WriterEvent.RESULT_STATUS],
                event[WriterEvent.RESULT_VALUE],
                self.project,
            )
        logger.info("Completed event DB writes")
