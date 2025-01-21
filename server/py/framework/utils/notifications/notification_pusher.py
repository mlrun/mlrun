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
import collections
import datetime
import traceback
import typing

from kubernetes.client import ApiException

import mlrun.common.schemas
import mlrun.errors
import mlrun.model
import mlrun.utils.helpers
import mlrun.utils.notifications.notification as notification_module
import mlrun.utils.notifications.notification.base as base
from mlrun.utils import Workflow, logger
from mlrun.utils.notifications.notification_pusher import (
    NotificationPusher,
    _NotificationPusherBase,
    sanitize_notification,
)

import framework.api.utils
import framework.constants
import framework.utils.singletons.k8s


class RunNotificationPusher(NotificationPusher):
    mail_notification_default_params = None

    @staticmethod
    def resolve_notifications_default_params():
        # TODO: After implementing make running notification send from the server side (ML-8069),
        #       we should move all the notifications classes from the client to the server and also
        #       create new function on the NotificationBase class for resolving the default params.
        #       After that we can remove this function.
        return {
            notification_module.NotificationTypes.console: {},
            notification_module.NotificationTypes.git: {},
            notification_module.NotificationTypes.ipython: {},
            notification_module.NotificationTypes.slack: {},
            notification_module.NotificationTypes.mail: RunNotificationPusher.get_mail_notification_default_params(),
            notification_module.NotificationTypes.webhook: {},
        }

    @staticmethod
    def get_mail_notification_default_params(refresh=False):
        # Check if the `refresh` flag is not set and the default mail notification parameters are already set.
        # Ensure `mail_notification_default_params` is not None or empty,
        # as an empty dictionary might indicate configuration changes we would like to reload.
        # This avoids unnecessary re-fetching unless a refresh is explicitly requested.
        if not refresh and RunNotificationPusher.mail_notification_default_params:
            return RunNotificationPusher.mail_notification_default_params

        mail_notification_default_params = (
            RunNotificationPusher._get_mail_notifications_default_params_from_secret()
        )

        RunNotificationPusher.mail_notification_default_params = (
            mail_notification_default_params
        )
        return RunNotificationPusher.mail_notification_default_params

    @staticmethod
    def _get_mail_notifications_default_params_from_secret():
        smtp_config_secret_name = mlrun.mlconf.notifications.smtp.config_secret_name
        mail_notification_default_params = {}
        if framework.utils.singletons.k8s.get_k8s_helper().is_running_inside_kubernetes_cluster():
            try:
                mail_notification_default_params = (
                    framework.utils.singletons.k8s.get_k8s_helper().read_secret_data(
                        smtp_config_secret_name, load_as_json=True, silent=True
                    )
                ) or {}
            except ApiException as exc:
                logger.warning(
                    "Failed to read SMTP configuration secret",
                    secret_name=smtp_config_secret_name,
                    body=mlrun.errors.err_to_str(exc.body),
                )
        return mail_notification_default_params

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
            framework.utils.notifications.mask_notification_params_on_task(
                run, framework.constants.MaskOperations.REDACT
            )

        return message, severity, runs


class AlertNotificationPusher(_NotificationPusherBase):
    def push(
        self,
        alert: mlrun.common.schemas.AlertConfig,
        event_data: mlrun.common.schemas.Event,
        activation_id: typing.Optional[int] = None,
        activation_time: typing.Optional[datetime.datetime] = None,
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
                    framework.utils.notifications.unmask_notification_params_secret(
                        alert.project, notification_object
                    )
                )

                name = notification_object.name
                notification_type = notification_module.NotificationTypes(
                    notification_object.kind
                )
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
            if activation_id and activation_time:
                # after notifications are sent, update the alert activation state
                # because only then the alert object had all the necessary data
                try:
                    self._update_alert_activation_notification_state(
                        activation_id=activation_id,
                        activation_time=activation_time,
                        alert=alert,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to update alert activation state", error=str(e)
                    )

        self._push(sync_push, async_push)

    def _update_alert_activation_notification_state(
        self,
        activation_id: int,
        activation_time: datetime.datetime,
        alert: mlrun.common.schemas.AlertConfig,
    ):
        normalized_activation_time = mlrun.utils.helpers.datetime_to_mysql_ts(
            activation_time
        )
        notification_states = self._prepare_notification_states(alert.notifications)
        db = mlrun.get_run_db()
        db.update_alert_activation(
            activation_id=activation_id,
            activation_time=normalized_activation_time,
            notifications_states=notification_states,
        )

    async def _push_notification_async(
        self,
        notification: base.NotificationBase,
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
                reason=str(exc),
            )
            raise exc

    @staticmethod
    def _prepare_notification_args(
        alert: mlrun.common.schemas.AlertConfig,
        notification_object: mlrun.common.schemas.Notification,
    ):
        message = (
            f": {notification_object.message}"
            if notification_object.message
            else alert.summary
        )

        severity = alert.severity
        return message, severity

    @staticmethod
    def _prepare_notification_states(
        notifications: list[mlrun.common.schemas.AlertNotification],
    ) -> list[mlrun.common.schemas.NotificationState]:
        """
        Processes a list of alert notifications to construct a list of NotificationState objects.

        Each NotificationState represents a unique type of notification (e.g., "slack", "email") and its status.
        For each notification type, this method aggregates error messages if any notifications of that type have failed.
        The resulting NotificationState has:
        - An empty 'err' if all notifications of that type succeeded.
        - An 'err' with all unique errors if all notifications of that type failed.
        - An 'err' with unique errors if some, but not all, notifications of that type failed.
        """

        notification_errors = collections.defaultdict(
            lambda: {
                "errors": set(),
                "success_count": 0,
                "failed_count": 0,
            },
        )

        # process each notification and gather errors by type
        for alert_notification in notifications:
            kind = alert_notification.notification.kind
            reason = alert_notification.notification.reason

            # count successes, failures and collect unique errors for failures
            if reason:
                notification_errors[kind]["errors"].add(reason)
                notification_errors[kind]["failed_count"] += 1
            else:
                notification_errors[kind]["success_count"] += 1

        # construct NotificationState objects based on the aggregated error data
        notification_states = []
        for kind, status_info in notification_errors.items():
            errors = list(status_info["errors"])
            success_count = status_info.get("success_count", 0)
            failed_count = status_info.get("failed_count", 0)

            if errors:
                if success_count == 0:
                    error_message = (
                        f"All {kind} notifications failed. Errors: {', '.join(errors)}"
                    )
                else:
                    error_message = (
                        f"Some {kind} notifications failed. Errors: {', '.join(errors)}"
                    )
            else:
                # indicates success if there are no errors
                error_message = ""

            notification_states.append(
                mlrun.common.schemas.NotificationState(
                    kind=kind,
                    err=error_message,
                    summary=mlrun.common.schemas.NotificationSummary(
                        failed=failed_count,
                        succeeded=success_count,
                    ),
                )
            )

        return notification_states

    @staticmethod
    def _update_notification_status(
        alert_id: int,
        project: str,
        notification: mlrun.common.schemas.Notification,
        status: typing.Optional[str] = None,
        sent_time: typing.Optional[datetime.datetime] = None,
        reason: typing.Optional[str] = None,
    ):
        db = mlrun.get_run_db()
        notification.status = status or notification.status
        notification.sent_time = sent_time or notification.sent_time

        # fill reason only if failed
        if notification.status == mlrun.common.schemas.NotificationStatus.ERROR:
            notification.reason = reason or notification.reason

            # limit reason to a max of 255 characters (for db reasons) but also for human readability reasons.
            notification.reason = notification.reason[:255]
        else:
            notification.reason = None

        # There is no need to mask the params as the secrets are already loaded
        db.store_alert_notifications(
            None,
            [notification],
            alert_id,
            project,
            mask_params=False,
        )


class KFPNotificationPusher(NotificationPusher):
    def __init__(
        self,
        project: str,
        workflow_id: str,
        notifications: list[mlrun.common.schemas.Notification],
        default_params: typing.Optional[dict] = None,
    ):
        self._project = project
        self._default_params = default_params or {}
        self._workflow_id = workflow_id
        self._notifications = notifications
        self._sync_notifications: list[
            tuple[base.NotificationBase, mlrun.model.Notification]
        ] = []
        self._async_notifications: list[
            tuple[base.NotificationBase, mlrun.model.Notification]
        ] = []

        for notification_object in self._notifications:
            try:
                notification = self._load_notification(notification_object)
                if notification.is_async:
                    self._async_notifications.append(
                        (notification, notification_object)
                    )
                else:
                    self._sync_notifications.append((notification, notification_object))
            except Exception as exc:
                logger.warning(
                    "Failed to process notification",
                    notification=notification_object.name,
                    error=mlrun.errors.err_to_str(exc),
                )

    def push(self, sync_push_callback=None, async_push_callback=None):
        def sync_push():
            for notification_data in self._sync_notifications:
                try:
                    self._push_workflow_notification_sync(
                        notification_data[0],
                        notification_data[1],
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to push notification sync",
                        error=mlrun.errors.err_to_str(exc),
                    )

        async def async_push():
            tasks = []
            for notification_data in self._async_notifications:
                tasks.append(
                    self._push_workflow_notification_async(
                        notification_data[0],
                        notification_data[1],
                    )
                )

            # return exceptions to "best-effort" fire all notifications
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(
                        "Failed to push notification async",
                        error=mlrun.errors.err_to_str(result),
                        traceback=traceback.format_exception(
                            result,
                            value=result,
                            tb=result.__traceback__,
                        ),
                    )

        super().push(sync_push, async_push)

    def _push_workflow_notification_sync(
        self,
        notification: base.NotificationBase,
        notification_object: mlrun.common.schemas.Notification,
    ):
        message, severity, runs = self._prepare_workflow_notification_args(
            notification_object
        )

        logger.debug(
            "Pushing sync notification",
            notification=sanitize_notification(notification_object.dict()),
            workflow_id=self._workflow_id,
        )
        try:
            notification.push(message, severity, runs)
            logger.debug(
                "Notification sent successfully",
                notification=sanitize_notification(notification_object.dict()),
                workflow_id=self._workflow_id,
            )
        except Exception as exc:
            logger.warning(
                "Failed to send or update notification",
                notification=sanitize_notification(notification_object.dict()),
                workflow_id=self._workflow_id,
                exc=mlrun.errors.err_to_str(exc),
                traceback=traceback.format_exc(),
            )
            raise exc

    async def _push_workflow_notification_async(
        self,
        notification: base.NotificationBase,
        notification_object: mlrun.common.schemas.Notification,
    ):
        message, severity, runs = self._prepare_workflow_notification_args(
            notification_object
        )

        logger.debug(
            "Pushing async notification",
            notification=sanitize_notification(notification_object.dict()),
            workflow_id=self._workflow_id,
        )
        try:
            await notification.push(message, severity, runs)
            logger.debug(
                "Notification sent successfully",
                notification=sanitize_notification(notification_object.dict()),
                workflow_id=self._workflow_id,
            )

        except Exception as exc:
            logger.warning(
                "Failed to send or update async notification",
                notification=sanitize_notification(notification_object.dict()),
                workflow_id=self._workflow_id,
                exc=mlrun.errors.err_to_str(exc),
                traceback=traceback.format_exc(),
            )

            raise exc

    def _prepare_workflow_notification_args(
        self, notification_object: mlrun.common.schemas.Notification
    ):
        custom_message = (
            f": {notification_object.message}" if notification_object.message else ""
        )

        message = f" (workflow: {self._workflow_id}){custom_message}"
        runs = Workflow.get_workflow_steps(self._workflow_id, self._project)

        severity = (
            notification_object.severity
            or mlrun.common.schemas.NotificationSeverity.INFO
        )
        return message, severity, runs
