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
from typing import Any, Callable, NewType

import mlrun.common.model_monitoring
import mlrun.common.schemas
import mlrun.common.schemas.alert as alert_objects
import mlrun.model_monitoring
from mlrun.common.schemas.model_monitoring.constants import (
    EventFieldType,
    HistogramDataDriftApplicationConstants,
    MetricData,
    ResultData,
    ResultKindApp,
    ResultStatusApp,
    WriterEvent,
    WriterEventKind,
)
from mlrun.common.schemas.notification import NotificationKind, NotificationSeverity
from mlrun.model_monitoring.helpers import get_result_instance_fqn
from mlrun.serving.utils import StepToDict
from mlrun.utils import logger
from mlrun.utils.notifications.notification_pusher import CustomNotificationPusher

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
        return self._event[ResultData.RESULT_STATUS] >= ResultStatusApp.detected.value

    def _generate_message(self) -> str:
        return f"""\
The monitoring app `{self._event[WriterEvent.APPLICATION_NAME]}` \
of kind `{self._event[ResultData.RESULT_KIND]}` \
detected a problem in model endpoint ID `{self._event[WriterEvent.ENDPOINT_ID]}` \
at time `{self._event[WriterEvent.START_INFER_TIME]}`.

Result data:
Name: `{self._event[ResultData.RESULT_NAME]}`
Value: `{self._event[ResultData.RESULT_VALUE]}`
Status: `{self._event[ResultData.RESULT_STATUS]}`
Extra data: `{self._event[ResultData.RESULT_EXTRA_DATA]}`\
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
    Write monitoring application results to the target databases
    """

    kind = "monitoring_application_stream_pusher"

    def __init__(
        self,
        project: str,
        secret_provider: Callable = None,
    ) -> None:
        self.project = project
        self.name = project  # required for the deployment process

        self._custom_notifier = CustomNotificationPusher(
            notification_types=[NotificationKind.slack]
        )

        self._app_result_store = mlrun.model_monitoring.get_store_object(
            project=self.project, secret_provider=secret_provider
        )
        self._tsdb_connector = mlrun.model_monitoring.get_tsdb_connector(
            project=self.project, secret_provider=secret_provider
        )
        self._endpoints_records = {}

    def _generate_event_on_drift(
        self,
        entity_id: str,
        result_status: int,
        event_value: dict,
        project_name: str,
        result_kind: int,
    ) -> None:
        entity = mlrun.common.schemas.alert.EventEntities(
            kind=alert_objects.EventEntityKind.MODEL_ENDPOINT_RESULT,
            project=project_name,
            ids=[entity_id],
        )

        event_kind = self._generate_alert_event_kind(
            result_status=result_status, result_kind=result_kind
        )

        event_data = mlrun.common.schemas.Event(
            kind=alert_objects.EventKind(value=event_kind),
            entity=entity,
            value_dict=event_value,
        )
        logger.info("Sending a drift event")
        mlrun.get_run_db().generate_event(event_kind, event_data)
        logger.info("Drift event sent successfully")

    @staticmethod
    def _generate_alert_event_kind(
        result_kind: int, result_status: int
    ) -> alert_objects.EventKind:
        """Generate the required Event Kind format for the alerting system"""
        event_kind = ResultKindApp(value=result_kind).name

        if result_status == ResultStatusApp.detected.value:
            event_kind = f"{event_kind}_detected"
        else:
            event_kind = f"{event_kind}_suspected"
        return alert_objects.EventKind(
            value=mlrun.utils.helpers.normalize_name(event_kind)
        )

    @staticmethod
    def _reconstruct_event(event: _RawEvent) -> tuple[_AppResultEvent, WriterEventKind]:
        """
        Modify the raw event into the expected monitoring application event
        schema as defined in `mlrun.common.schemas.model_monitoring.constants.WriterEvent`
        """
        if not isinstance(event, dict):
            raise _WriterEventTypeError(
                f"The event is of type: {type(event)}, expected a dictionary"
            )
        kind = event.pop(WriterEvent.EVENT_KIND, WriterEventKind.RESULT)
        result_event = _AppResultEvent(json.loads(event.pop(WriterEvent.DATA, "{}")))
        if not result_event:  # BC for < 1.7.0, can be removed in 1.9.0
            result_event = _AppResultEvent(event)
        else:
            result_event.update(_AppResultEvent(event))

        expected_keys = list(
            set(WriterEvent.list()).difference(
                [WriterEvent.EVENT_KIND, WriterEvent.DATA]
            )
        )
        if kind == WriterEventKind.METRIC:
            expected_keys.extend(MetricData.list())
        elif kind == WriterEventKind.RESULT:
            expected_keys.extend(ResultData.list())
        else:
            raise _WriterEventValueError(
                f"Unknown event kind: {kind}, expected one of: {WriterEventKind.list()}"
            )
        missing_keys = [key for key in expected_keys if key not in result_event]
        if missing_keys:
            raise _WriterEventValueError(
                f"The received event misses some keys compared to the expected "
                f"monitoring application event schema: {missing_keys}"
            )

        return result_event, kind

    def do(self, event: _RawEvent) -> None:
        event, kind = self._reconstruct_event(event)
        logger.info("Starting to write event", event=event)
        self._tsdb_connector.write_application_event(event=event.copy(), kind=kind)
        self._app_result_store.write_application_event(event=event.copy(), kind=kind)

        logger.info("Completed event DB writes")

        if kind == WriterEventKind.RESULT:
            _Notifier(event=event, notification_pusher=self._custom_notifier).notify()

        if (
            mlrun.mlconf.alerts.mode == mlrun.common.schemas.alert.AlertsModes.enabled
            and kind == WriterEventKind.RESULT
            and (
                event[ResultData.RESULT_STATUS] == ResultStatusApp.detected.value
                or event[ResultData.RESULT_STATUS]
                == ResultStatusApp.potential_detection.value
            )
        ):
            endpoint_id = event[WriterEvent.ENDPOINT_ID]
            endpoint_record = self._endpoints_records.setdefault(
                endpoint_id,
                self._app_result_store.get_model_endpoint(endpoint_id=endpoint_id),
            )
            event_value = {
                "app_name": event[WriterEvent.APPLICATION_NAME],
                "model": endpoint_record.get(EventFieldType.MODEL),
                "model_endpoint_id": event[WriterEvent.ENDPOINT_ID],
                "result_name": event[ResultData.RESULT_NAME],
                "result_value": event[ResultData.RESULT_VALUE],
            }
            self._generate_event_on_drift(
                entity_id=get_result_instance_fqn(
                    event[WriterEvent.ENDPOINT_ID],
                    event[WriterEvent.APPLICATION_NAME],
                    event[ResultData.RESULT_NAME],
                ),
                result_status=event[ResultData.RESULT_STATUS],
                event_value=event_value,
                project_name=self.project,
                result_kind=event[ResultData.RESULT_KIND],
            )

        if (
            kind == WriterEventKind.RESULT
            and event[WriterEvent.APPLICATION_NAME]
            == HistogramDataDriftApplicationConstants.NAME
            and event[ResultData.RESULT_NAME]
            == HistogramDataDriftApplicationConstants.GENERAL_RESULT_NAME
        ):
            endpoint_id = event[WriterEvent.ENDPOINT_ID]
            logger.info(
                "Updating the model endpoint with metadata specific to the histogram "
                "data drift app",
                endpoint_id=endpoint_id,
            )
            attributes = json.loads(event[ResultData.RESULT_EXTRA_DATA])
            attributes[EventFieldType.DRIFT_STATUS] = str(
                attributes[EventFieldType.DRIFT_STATUS]
            )
            self._app_result_store.update_model_endpoint(
                endpoint_id=endpoint_id,
                attributes=attributes,
            )

        logger.info("Model monitoring writer finished handling event")
