# Copyright 2018 Iguazio
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
import os
import typing

from fastapi.concurrency import run_in_threadpool

import mlrun.api.db.base
import mlrun.api.db.session
import mlrun.common.schemas
import mlrun.config
import mlrun.lists
import mlrun.model
import mlrun.utils.helpers
from mlrun.utils import logger
from mlrun.utils.condition_evaluator import evaluate_condition_in_separate_process

from .notification import NotificationBase, NotificationTypes


class NotificationPusher(object):

    messages = {
        "completed": "Run completed",
        "error": "Run failed",
        "aborted": "Run aborted",
    }

    def __init__(self, runs: typing.Union[mlrun.lists.RunList, list]):
        self._runs = runs
        self._sync_notifications = []
        self._async_notifications = []

        for run in self._runs:
            if isinstance(run, dict):
                run = mlrun.model.RunObject.from_dict(run)

            for notification in run.spec.notifications:
                try:
                    notification.status = run.status.notifications.get(
                        notification.name
                    ).get("status", mlrun.common.schemas.NotificationStatus.PENDING)
                except (AttributeError, KeyError):
                    notification.status = (
                        mlrun.common.schemas.NotificationStatus.PENDING
                    )

                if self._should_notify(run, notification):
                    self._load_notification(run, notification)

    def push(
        self,
        db: mlrun.api.db.base.DBInterface = None,
    ):
        """
        Asynchronously push notifications for all runs in the initialized runs list (if they should be pushed).
        When running from a sync environment, the notifications will be pushed asynchronously however the function will
        wait for all notifications to be pushed before returning.
        """

        if not len(self._sync_notifications) and not len(self._async_notifications):
            return

        def _sync_push():
            for notification_data in self._sync_notifications:
                self._push_notification_sync(
                    notification_data[0],
                    notification_data[1],
                    notification_data[2],
                    db,
                )

        async def _async_push():
            tasks = []
            for notification_data in self._async_notifications:
                tasks.append(
                    self._push_notification_async(
                        notification_data[0],
                        notification_data[1],
                        notification_data[2],
                        db,
                    )
                )
            await asyncio.gather(*tasks)

        logger.debug(
            "Pushing notifications",
            notifications_amount=len(self._sync_notifications)
            + len(self._async_notifications),
        )

        # first push async notifications
        main_event_loop = asyncio.get_event_loop()
        if main_event_loop.is_running():

            # If running from the api or from jupyter notebook, we are already in an event loop.
            # We add the async push function to the loop and run it.
            asyncio.run_coroutine_threadsafe(_async_push(), main_event_loop)
        else:

            # If running mlrun SDK locally (not from jupyter), there isn't necessarily an event loop.
            # We create a new event loop and run the async push function in it.
            main_event_loop.run_until_complete(_async_push())

        # then push sync notifications
        if not mlrun.config.is_running_as_api():
            _sync_push()

    @staticmethod
    def _should_notify(
        run: mlrun.model.RunObject,
        notification: mlrun.model.Notification,
    ) -> bool:
        when_states = notification.when
        run_state = run.state()

        # if the notification isn't pending, don't push it
        if (
            notification.status
            and notification.status != mlrun.common.schemas.NotificationStatus.PENDING
        ):
            return False

        # if at least one condition is met, notify
        for when_state in when_states:
            if when_state == run_state:
                if (
                    run_state == "completed"
                    and evaluate_condition_in_separate_process(
                        notification.condition,
                        context={
                            "run": run.to_dict(),
                            "notification": notification.to_dict(),
                        },
                    )
                ) or run_state in ["error", "aborted"]:
                    return True

        return False

    def _load_notification(
        self, run: mlrun.model.RunObject, notification_object: mlrun.model.Notification
    ) -> NotificationBase:
        name = notification_object.name
        notification_type = NotificationTypes(
            notification_object.kind or NotificationTypes.console
        )
        notification = notification_type.get_notification()(
            name, notification_object.params
        )
        if notification.is_async:
            self._async_notifications.append((notification, run, notification_object))
        else:
            self._sync_notifications.append((notification, run, notification_object))

        logger.debug(
            "Loaded notification", notification=name, type=notification_type.value
        )
        return notification

    def _prepare_notification_args(
        self, run: mlrun.model.RunObject, notification_object: mlrun.model.Notification
    ):
        custom_message = (
            f": {notification_object.message}" if notification_object.message else ""
        )
        message = self.messages.get(run.state(), "") + custom_message

        severity = (
            notification_object.severity
            or mlrun.common.schemas.NotificationSeverity.INFO
        )
        return message, severity, [run.to_dict()]

    def _push_notification_sync(
        self,
        notification: NotificationBase,
        run: mlrun.model.RunObject,
        notification_object: mlrun.model.Notification,
        db: mlrun.api.db.base.DBInterface,
    ):
        message, severity, runs = self._prepare_notification_args(
            run, notification_object
        )
        logger.debug(
            "Pushing sync notification",
            notification=_sanitize_notification(notification_object),
            run_uid=run.metadata.uid,
        )
        try:
            notification.push(message, severity, runs)
            self._update_notification_status(
                db,
                run.metadata.uid,
                run.metadata.project,
                notification_object,
                status=mlrun.common.schemas.NotificationStatus.SENT,
                sent_time=datetime.datetime.now(tz=datetime.timezone.utc),
            )
        except Exception as exc:
            self._update_notification_status(
                db,
                run.metadata.uid,
                run.metadata.project,
                notification_object,
                status=mlrun.common.schemas.NotificationStatus.ERROR,
            )
            raise exc

    async def _push_notification_async(
        self,
        notification: NotificationBase,
        run: mlrun.model.RunObject,
        notification_object: mlrun.model.Notification,
        db: mlrun.api.db.base.DBInterface,
    ):
        message, severity, runs = self._prepare_notification_args(
            run, notification_object
        )
        logger.debug(
            "Pushing async notification",
            notification=_sanitize_notification(notification_object),
            run_uid=run.metadata.uid,
        )
        try:
            await notification.push(message, severity, runs)

            await run_in_threadpool(
                self._update_notification_status,
                db,
                run.metadata.uid,
                run.metadata.project,
                notification_object,
                status=mlrun.common.schemas.NotificationStatus.SENT,
                sent_time=datetime.datetime.now(tz=datetime.timezone.utc),
            )
        except Exception as exc:
            await run_in_threadpool(
                self._update_notification_status,
                db,
                run.metadata.uid,
                run.metadata.project,
                notification_object,
                status=mlrun.common.schemas.NotificationStatus.ERROR,
            )
            raise exc

    @staticmethod
    def _update_notification_status(
        db: mlrun.api.db.base.DBInterface,
        run_uid: str,
        project: str,
        notification: mlrun.model.Notification,
        status: str = None,
        sent_time: typing.Optional[datetime.datetime] = None,
    ):

        # nothing to update if not running as api
        # note, the notification mechanism may run "locally" for certain runtimes
        if not mlrun.config.is_running_as_api():
            return

        # TODO: move to api side
        db_session = mlrun.api.db.session.create_session()
        notification.status = status or notification.status
        notification.sent_time = sent_time or notification.sent_time

        # store directly in db, no need to use crud as the secrets are already loaded
        db.store_run_notifications(
            db_session,
            [notification],
            run_uid,
            project,
        )


class CustomNotificationPusher(object):
    def __init__(self, notification_types: typing.List[str] = None):
        notifications = {
            notification_type: NotificationTypes(notification_type).get_notification()()
            for notification_type in notification_types
        }
        self._sync_notifications = {
            notification_type: notification
            for notification_type, notification in notifications.items()
            if not notification.is_async
        }
        self._async_notifications = {
            notification_type: notification
            for notification_type, notification in notifications.items()
            if notification.is_async
        }

    def push(
        self,
        message: str,
        severity: typing.Union[
            mlrun.common.schemas.NotificationSeverity, str
        ] = mlrun.common.schemas.NotificationSeverity.INFO,
        runs: typing.Union[mlrun.lists.RunList, list] = None,
        custom_html: str = None,
    ):
        def _sync_push():
            for notification_type, notification in self._sync_notifications.items():
                if self.should_push_notification(notification_type):
                    notification.push(message, severity, runs, custom_html)

        async def _async_push():
            tasks = []
            for notification_type, notification in self._async_notifications.items():
                if self.should_push_notification(notification_type):
                    tasks.append(
                        notification.push(message, severity, runs, custom_html)
                    )
            await asyncio.gather(*tasks)

        # first push async notifications
        main_event_loop = asyncio.get_event_loop()
        if main_event_loop.is_running():
            asyncio.run_coroutine_threadsafe(_async_push(), main_event_loop)
        else:
            main_event_loop.run_until_complete(_async_push())

        # then push sync notifications
        if not mlrun.config.is_running_as_api():
            _sync_push()

    def add_notification(
        self, notification_type: str, params: typing.Dict[str, str] = None
    ):
        if notification_type in self._async_notifications:
            self._async_notifications[notification_type].load_notification(params)
        elif notification_type in self._sync_notifications:
            self._sync_notifications[notification_type].load_notification(params)
        else:
            notification = NotificationTypes(notification_type).get_notification()(
                params
            )
            if notification.is_async:
                self._async_notifications[notification_type] = notification
            else:
                self._sync_notifications[notification_type] = notification

    def should_push_notification(self, notification_type):
        notifications = {}
        notifications.update(self._sync_notifications)
        notifications.update(self._async_notifications)
        notification = notifications.get(notification_type)
        if not notification or not notification.active:
            return False

        # get notification's inverse dependencies, and only push the notification if
        # none of its inverse dependencies are being sent
        inverse_dependencies = NotificationTypes(
            notification_type
        ).inverse_dependencies()
        for inverse_dependency in inverse_dependencies:
            inverse_dependency_notification = notifications.get(inverse_dependency)
            if (
                inverse_dependency_notification
                and inverse_dependency_notification.active
            ):
                return False

        return True

    def push_pipeline_start_message(
        self,
        project: str,
        commit_id: str = None,
        pipeline_id: str = None,
        has_workflow_url: bool = False,
    ):
        message = f"Pipeline started in project {project}"
        if pipeline_id:
            message += f" id={pipeline_id}"
        commit_id = (
            commit_id or os.environ.get("GITHUB_SHA") or os.environ.get("CI_COMMIT_SHA")
        )
        if commit_id:
            message += f", commit={commit_id}"
        if has_workflow_url:
            url = mlrun.utils.helpers.get_workflow_url(project, pipeline_id)
        else:
            url = mlrun.utils.helpers.get_ui_url(project)
        html = ""
        if url:
            html = (
                message
                + f'<div><a href="{url}" target="_blank">click here to view progress</a></div>'
            )
            message = message + f", check progress in {url}"
        self.push(message, "info", custom_html=html)

    def push_pipeline_run_results(
        self,
        runs: typing.Union[mlrun.lists.RunList, list],
        push_all: bool = False,
        state: str = None,
    ):
        """
        push a structured table with run results to notification targets

        :param runs:  list if run objects (RunObject)
        :param push_all: push all notifications (including already notified runs)
        :param state: final run state
        """
        had_errors = 0
        runs_list = []
        for run in runs:
            notified = getattr(run, "_notified", False)
            if not notified or push_all:
                if run.status.state == "error":
                    had_errors += 1
                runs_list.append(run.to_dict())
                run._notified = True

        text = "pipeline run finished"
        if had_errors:
            text += f" with {had_errors} errors"
        if state:
            text += f", state={state}"
        self.push(text, "info", runs=runs_list)


def _sanitize_notification(notification: mlrun.model.Notification):
    notification_dict = notification.to_dict()
    notification_dict.pop("params", None)
    return notification_dict


def _separate_sync_notifications(
    notifications: typing.List[NotificationBase],
) -> typing.Tuple[typing.List[NotificationBase], typing.List[NotificationBase]]:
    sync_notifications = []
    async_notifications = []
    for notification in notifications:
        if notification.is_async:
            async_notifications.append(notification)
        else:
            sync_notifications.append(notification)
    return sync_notifications, async_notifications
