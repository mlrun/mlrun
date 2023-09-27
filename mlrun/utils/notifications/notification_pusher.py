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
import os
import traceback
import typing
from concurrent.futures import ThreadPoolExecutor

import mlrun.common.schemas
import mlrun.config
import mlrun.db.base
import mlrun.errors
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
        self._sync_notifications: typing.List[
            typing.Tuple[
                NotificationBase, mlrun.model.RunObject, mlrun.model.Notification
            ]
        ] = []
        self._async_notifications: typing.List[
            typing.Tuple[
                NotificationBase, mlrun.model.RunObject, mlrun.model.Notification
            ]
        ] = []

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

    def push(self):
        """
        Asynchronously push notifications for all runs in the initialized runs list (if they should be pushed).
        When running from a sync environment, the notifications will be pushed asynchronously however the function will
        wait for all notifications to be pushed before returning.
        """

        if not len(self._sync_notifications) and not len(self._async_notifications):
            return

        def _sync_push():
            for notification_data in self._sync_notifications:
                try:
                    self._push_notification_sync(
                        notification_data[0],
                        notification_data[1],
                        notification_data[2],
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to push notification sync",
                        error=mlrun.errors.err_to_str(exc),
                    )

        async def _async_push():
            tasks = []
            for notification_data in self._async_notifications:
                tasks.append(
                    self._push_notification_async(
                        notification_data[0],
                        notification_data[1],
                        notification_data[2],
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

        logger.debug(
            "Pushing notifications",
            notifications_amount=len(self._sync_notifications)
            + len(self._async_notifications),
        )

        # first push async notifications
        main_event_loop = asyncio.get_event_loop()
        if not main_event_loop.is_running():
            # If running mlrun SDK locally (not from jupyter), there isn't necessarily an event loop.
            # We create a new event loop and run the async push function in it.
            main_event_loop.run_until_complete(_async_push())
        elif mlrun.utils.helpers.is_running_in_jupyter_notebook():
            # Running in Jupyter notebook.
            # In this case, we need to create a new thread, run a separate event loop in
            # that thread, and use it instead of the main_event_loop.
            # This is necessary because Jupyter Notebook has its own event loop,
            # but it runs in the main thread. As long as a cell is running,
            # the event loop will not execute properly
            _run_coroutine_in_jupyter_notebook(coroutine_method=_async_push)
        else:
            # Running in mlrun api, we are in a separate thread from the one in which
            # the main event loop, so we can just send the notifications to that loop
            asyncio.run_coroutine_threadsafe(_async_push(), main_event_loop)

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
        params = {}
        params.update(notification_object.secret_params)
        params.update(notification_object.params)
        notification = notification_type.get_notification()(name, params)
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
    ):
        message, severity, runs = self._prepare_notification_args(
            run, notification_object
        )
        logger.debug(
            "Pushing sync notification",
            notification=sanitize_notification(notification_object.to_dict()),
            run_uid=run.metadata.uid,
        )
        update_notification_status_kwargs = {
            "run_uid": run.metadata.uid,
            "project": run.metadata.project,
            "notification": notification_object,
            "status": mlrun.common.schemas.NotificationStatus.SENT,
        }
        try:
            notification.push(message, severity, runs)
            logger.debug(
                "Notification sent successfully",
                notification=sanitize_notification(notification_object.to_dict()),
                run_uid=run.metadata.uid,
            )
            update_notification_status_kwargs["sent_time"] = datetime.datetime.now(
                tz=datetime.timezone.utc
            )
        except Exception as exc:
            logger.warning(
                "Failed to send or update notification",
                notification=sanitize_notification(notification_object.to_dict()),
                run_uid=run.metadata.uid,
                exc=mlrun.errors.err_to_str(exc),
                traceback=traceback.format_exc(),
            )
            update_notification_status_kwargs["reason"] = f"Exception error: {str(exc)}"
            update_notification_status_kwargs[
                "status"
            ] = mlrun.common.schemas.NotificationStatus.ERROR
            raise exc
        finally:
            self._update_notification_status(
                **update_notification_status_kwargs,
            )

    async def _push_notification_async(
        self,
        notification: NotificationBase,
        run: mlrun.model.RunObject,
        notification_object: mlrun.model.Notification,
    ):
        message, severity, runs = self._prepare_notification_args(
            run, notification_object
        )
        logger.debug(
            "Pushing async notification",
            notification=sanitize_notification(notification_object.to_dict()),
            run_uid=run.metadata.uid,
        )
        update_notification_status_kwargs = {
            "run_uid": run.metadata.uid,
            "project": run.metadata.project,
            "notification": notification_object,
            "status": mlrun.common.schemas.NotificationStatus.SENT,
        }
        try:
            await notification.push(message, severity, runs)
            logger.debug(
                "Notification sent successfully",
                notification=sanitize_notification(notification_object.to_dict()),
                run_uid=run.metadata.uid,
            )
            update_notification_status_kwargs["sent_time"] = datetime.datetime.now(
                tz=datetime.timezone.utc
            )

        except Exception as exc:
            logger.warning(
                "Failed to send or update notification",
                notification=sanitize_notification(notification_object.to_dict()),
                run_uid=run.metadata.uid,
                exc=mlrun.errors.err_to_str(exc),
                traceback=traceback.format_exc(),
            )
            update_notification_status_kwargs["reason"] = f"Exception error: {str(exc)}"
            update_notification_status_kwargs[
                "status"
            ] = mlrun.common.schemas.NotificationStatus.ERROR
            raise exc
        finally:
            await mlrun.utils.helpers.run_in_threadpool(
                self._update_notification_status,
                **update_notification_status_kwargs,
            )

    @staticmethod
    def _update_notification_status(
        run_uid: str,
        project: str,
        notification: mlrun.model.Notification,
        status: str = None,
        sent_time: typing.Optional[datetime.datetime] = None,
        reason: typing.Optional[str] = None,
    ):
        db = mlrun.get_run_db()
        notification.status = status or notification.status
        notification.sent_time = sent_time or notification.sent_time

        # fill reason only if failed
        if notification.status == mlrun.common.schemas.NotificationStatus.ERROR:
            notification.reason = reason or notification.reason

            # limit reason to a max of 255 characters (for db reasons)
            # but also for human readability reasons.
            notification.reason = notification.reason[:255]
        else:

            # empty out the reason if the notification is in a non-error state
            # in case a retry would kick in (when such mechanism would be implemented)
            notification.reason = None

        # There is no need to mask the secret_params as the secrets are already loaded
        db.store_run_notifications(
            [notification],
            run_uid,
            project,
            mask_params=False,
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
            # return exceptions to "best-effort" fire all notifications
            await asyncio.gather(*tasks, return_exceptions=True)

        # first push async notifications
        main_event_loop = asyncio.get_event_loop()
        if not main_event_loop.is_running():
            # If running mlrun SDK locally (not from jupyter), there isn't necessarily an event loop.
            # We create a new event loop and run the async push function in it.
            main_event_loop.run_until_complete(_async_push())
        elif mlrun.utils.helpers.is_running_in_jupyter_notebook():
            # Running in Jupyter notebook.
            # In this case, we need to create a new thread, run a separate event loop in
            # that thread, and use it instead of the main_event_loop.
            # This is necessary because Jupyter Notebook has its own event loop,
            # but it runs in the main thread. As long as a cell is running,
            # the event loop will not execute properly
            _run_coroutine_in_jupyter_notebook(coroutine_method=_async_push)
        else:
            # Running in mlrun api, we are in a separate thread from the one in which
            # the main event loop, so we can just send the notifications to that loop
            asyncio.run_coroutine_threadsafe(_async_push(), main_event_loop)

        # then push sync notifications
        if not mlrun.config.is_running_as_api():
            _sync_push()

    def add_notification(
        self,
        notification_type: str,
        params: typing.Dict[str, str] = None,
    ):
        if notification_type in self._async_notifications:
            self._async_notifications[notification_type].load_notification(params)
        elif notification_type in self._sync_notifications:
            self._sync_notifications[notification_type].load_notification(params)
        else:
            notification = NotificationTypes(notification_type).get_notification()(
                params=params,
            )
            if notification.is_async:
                self._async_notifications[notification_type] = notification
            else:
                self._sync_notifications[notification_type] = notification

    def remove_notification(self, notification_type: str):
        if notification_type in self._async_notifications:
            del self._async_notifications[notification_type]

        elif notification_type in self._sync_notifications:
            del self._sync_notifications[notification_type]

        else:
            logger.warning(f"No notification of type {notification_type} in project")

    def edit_notification(
        self, notification_type: str, params: typing.Dict[str, str] = None
    ):
        self.remove_notification(notification_type)
        self.add_notification(notification_type, params)

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
        message = f"Workflow started in project {project}"
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


def sanitize_notification(notification_dict: dict):
    notification_dict.pop("secret_params", None)
    notification_dict.pop("message", None)
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


def _run_coroutine_in_jupyter_notebook(coroutine_method):
    """
    Execute a coroutine in a Jupyter Notebook environment.

    This function creates a new thread pool executor with a single thread and a new event loop.
    It sets the created event loop as the current event loop.
    Then, it submits the coroutine to the event loop and waits for its completion.

    This approach is used in Jupyter Notebook to ensure the proper execution of the event loop in a separate thread,
    allowing for the asynchronous push operation to be executed while the notebook is running.

    :param coroutine_method: The coroutine method to be executed.
    :return: The result of the executed coroutine.
    """
    thread_pool_executer = ThreadPoolExecutor(1)
    async_event_loop = asyncio.new_event_loop()
    thread_pool_executer.submit(asyncio.set_event_loop, async_event_loop).result()
    result = thread_pool_executer.submit(
        async_event_loop.run_until_complete, coroutine_method()
    ).result()
    return result
