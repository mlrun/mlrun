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

import asyncio
import datetime
import typing

import mlrun.common.schemas
import mlrun.errors
import mlrun.model
import mlrun.utils.helpers
import server.api.api.utils
import server.api.constants
from mlrun.utils import logger
from mlrun.utils.notifications.notification import NotificationBase, NotificationTypes
from mlrun.utils.notifications.notification_pusher import (
    NotificationPusher,
    _NotificationPusherBase,
)


class RunNotificationPusher(NotificationPusher):
    def _prepare_notification_args(
        self, run: mlrun.model.RunObject, notification_object: mlrun.model.Notification
    ):
        """
        Prepare notification arguments for the notification pusher.
        In the server side implementation, we need to mask the notification parameters on the task as they are
        unmasked to extract the credentials required to send the notification.
        """
        message, severity, runs = super()._prepare_notification_args(
            run, notification_object
        )
        for run in runs:
            server.api.api.utils.mask_notification_params_on_task(
                run, server.api.constants.MaskOperations.REDACT
            )

        return message, severity, runs


class AlertNotificationPusher(_NotificationPusherBase):
    def push(
        self,
        alert: mlrun.common.schemas.AlertConfig,
        event_data: mlrun.common.schemas.Event,
    ):
        """
        Asynchronously push notification.
        """

        def sync_push():
            pass

        async def async_push():
            tasks = []
            for notification_data in alert.notifications:
                notification_object = mlrun.model.Notification.from_dict(
                    notification_data.notification.dict()
                )

                notification_object = (
                    server.api.api.utils.unmask_notification_params_secret(
                        alert.project, notification_object
                    )
                )

                name = notification_object.name
                notification_type = NotificationTypes(notification_object.kind)
                params = {}
                if notification_object.secret_params:
                    params.update(notification_object.secret_params)
                if notification_object.params:
                    params.update(notification_object.params)
                notification = notification_type.get_notification()(name, params)

                tasks.append(
                    self._push_notification_async(
                        notification,
                        alert,
                        notification_data.notification,
                        event_data,
                    )
                )

            # return exceptions to "best-effort" fire all notifications
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(
                        "Failed to push notification async",
                        error=mlrun.errors.err_to_str(result),
                    )

        self._push(sync_push, async_push)

    async def _push_notification_async(
        self,
        notification: NotificationBase,
        alert: mlrun.common.schemas.AlertConfig,
        notification_object: mlrun.common.schemas.Notification,
        event_data: mlrun.common.schemas.Event,
    ):
        message, severity = self._prepare_notification_args(alert, notification_object)
        logger.debug(
            "Pushing async notification",
            notification=notification_object,
            name=alert.name,
        )
        try:
            await notification.push(
                message, severity, alert=alert, event_data=event_data
            )
            logger.debug(
                "Notification sent successfully",
                notification=notification_object,
                name=alert.name,
            )
            await mlrun.utils.helpers.run_in_threadpool(
                self._update_notification_status,
                alert.id,
                alert.project,
                notification_object,
                status=mlrun.common.schemas.NotificationStatus.SENT,
                sent_time=datetime.datetime.now(tz=datetime.timezone.utc),
            )
        except Exception as exc:
            logger.warning(
                "Failed to send notification",
                notification=notification_object,
                name=alert.name,
                exc=mlrun.errors.err_to_str(exc),
            )
            await mlrun.utils.helpers.run_in_threadpool(
                self._update_notification_status,
                alert.id,
                alert.project,
                notification_object,
                status=mlrun.common.schemas.NotificationStatus.ERROR,
            )
            raise exc

    @staticmethod
    def _prepare_notification_args(
        alert: mlrun.common.schemas.AlertConfig,
        notification_object: mlrun.common.schemas.Notification,
    ):
        message = (
            f": {notification_object.message}" if notification_object.message else ""
        )

        severity = (
            notification_object.severity
            or mlrun.common.schemas.NotificationSeverity.INFO
        )
        return message, severity

    @staticmethod
    def _update_notification_status(
        alert_id: int,
        project: str,
        notification: mlrun.common.schemas.Notification,
        status: str = None,
        sent_time: typing.Optional[datetime.datetime] = None,
    ):
        db = mlrun.get_run_db()
        notification.status = status or notification.status
        notification.sent_time = sent_time or notification.sent_time

        # There is no need to mask the params as the secrets are already loaded
        db.store_alert_notifications(
            None,
            [notification],
            alert_id,
            project,
            mask_params=False,
        )
