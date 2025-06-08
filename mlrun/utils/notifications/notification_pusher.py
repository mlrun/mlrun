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

import mlrun.common.constants as mlrun_constants
import mlrun.common.runtimes.constants as runtimes_constants
import mlrun.common.schemas
import mlrun.config
import mlrun.db.base
import mlrun.errors
import mlrun.lists
import mlrun.model
import mlrun.utils.helpers
import mlrun.utils.notifications.notification as notification_module
import mlrun.utils.notifications.notification.base as base
from mlrun.utils import Workflow, logger
from mlrun.utils.condition_evaluator import evaluate_condition_in_separate_process


class _NotificationPusherBase:
    def _push(
        self, sync_push_callback: typing.Callable, async_push_callback: typing.Callable
    ):
        if mlrun.utils.helpers.is_running_in_jupyter_notebook():
            # Running in Jupyter notebook.
            # In this case, we need to create a new thread, run a separate event loop in
            # that thread, and use it instead of the main_event_loop.
            # This is necessary because Jupyter Notebook has its own event loop,
            # but it runs in the main thread. As long as a cell is running,
            # the event loop will not execute properly
            self._run_coroutine_in_jupyter_notebook(
                coroutine_method=async_push_callback
            )
        else:
            # Either running in mlrun api or sdk. in case of mlrun api we are in a separated thread, thus creating
            # a new event loop. in case of sdk, we are most likely in main thread, thus using the main event loop.
            try:
                event_loop = asyncio.get_event_loop()
            except RuntimeError:
                event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(event_loop)

            if not event_loop.is_running():
                event_loop.run_until_complete(async_push_callback())
            else:
                asyncio.run_coroutine_threadsafe(async_push_callback(), event_loop)

        # then push sync notifications
        if not mlrun.config.is_running_as_api():
            sync_push_callback()

    def _run_coroutine_in_jupyter_notebook(self, coroutine_method):
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
        thread_pool_executor = ThreadPoolExecutor(1)
        async_event_loop = asyncio.new_event_loop()
        thread_pool_executor.submit(asyncio.set_event_loop, async_event_loop).result()
        result = thread_pool_executor.submit(
            async_event_loop.run_until_complete, coroutine_method()
        ).result()
        return result


class NotificationPusher(_NotificationPusherBase):
    messages = {
        "completed": "{resource} completed",
        "error": "{resource} failed",
        "aborted": "{resource} aborted",
        "running": "{resource} started",
    }

    def __init__(
        self,
        runs: typing.Union[mlrun.lists.RunList, list],
        default_params: typing.Optional[dict] = None,
    ):
        self._runs = runs
        self._default_params = default_params or {}
        self._sync_notifications: list[
            tuple[
                base.NotificationBase, mlrun.model.RunObject, mlrun.model.Notification
            ]
        ] = []
        self._async_notifications: list[
            tuple[
                base.NotificationBase, mlrun.model.RunObject, mlrun.model.Notification
            ]
        ] = []

        for run in self._runs:
            try:
                self._process_run(run)
            except Exception as exc:
                logger.warning(
                    "Failed to process run",
                    run_uid=run.metadata.uid,
                    error=mlrun.errors.err_to_str(exc),
                )

    def _process_run(self, run):
        if isinstance(run, dict):
            run = mlrun.model.RunObject.from_dict(run)

        for notification in run.spec.notifications:
            try:
                self._process_notification(notification, run)
            except Exception as exc:
                logger.warning(
                    "Failed to process notification",
                    run_uid=run.metadata.uid,
                    notification=notification,
                    error=mlrun.errors.err_to_str(exc),
                )

    def _process_notification(self, notification_object, run):
        notification_object.status = run.status.notifications.get(
            notification_object.name, {}
        ).get(
            "status",
            mlrun.common.schemas.NotificationStatus.PENDING,
        )
        if self._should_notify(run, notification_object):
            notification = self._load_notification(notification_object)
            if notification.is_async:
                self._async_notifications.append(
                    (notification, run, notification_object)
                )
            else:
                self._sync_notifications.append(
                    (notification, run, notification_object)
                )

    def push(self, sync_push_callback=None, async_push_callback=None):
        """
        Asynchronously push notifications for all runs in the initialized runs list (if they should be pushed).
        When running from a sync environment, the notifications will be pushed asynchronously however the function will
        wait for all notifications to be pushed before returning.
        """

        if not len(self._sync_notifications) and not len(self._async_notifications):
            return

        def sync_push():
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

        async def async_push():
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
                        traceback=traceback.format_exception(
                            result,
                            value=result,
                            tb=result.__traceback__,
                        ),
                    )

        logger.debug(
            "Pushing notifications",
            notifications_amount=len(self._sync_notifications)
            + len(self._async_notifications),
        )
        sync_push_callback = sync_push_callback or sync_push
        async_push_callback = async_push_callback or async_push
        self._push(sync_push_callback, async_push_callback)

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
                    run_state == runtimes_constants.RunStates.completed
                    and evaluate_condition_in_separate_process(
                        notification.condition,
                        context={
                            "run": run.to_dict(),
                            "notification": notification.to_dict(),
                        },
                    )
                ) or run_state in [
                    runtimes_constants.RunStates.error,
                    runtimes_constants.RunStates.aborted,
                    runtimes_constants.RunStates.running,
                ]:
                    return True

        return False

    def _load_notification(
        self, notification_object: mlrun.model.Notification
    ) -> base.NotificationBase:
        name = notification_object.name
        notification_type = notification_module.NotificationTypes(
            notification_object.kind or notification_module.NotificationTypes.console
        )
        params = {}
        params.update(notification_object.secret_params or {})
        params.update(notification_object.params or {})
        default_params = self._default_params.get(notification_type.value, {})
        notification = notification_type.get_notification()(
            name, params, default_params
        )
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
        resource = "Run"
        runs = [run.to_dict()]

        if mlrun_constants.MLRunInternalLabels.workflow in run.metadata.labels:
            resource = mlrun_constants.MLRunInternalLabels.workflow
            custom_message = (
                f" (workflow: {run.metadata.labels['workflow']}){custom_message}"
            )
            project = run.metadata.project
            workflow_id = run.status.results.get("workflow_id", None)
            db = mlrun.get_run_db()
            runs.extend(Workflow.get_workflow_steps(db, workflow_id, project))

        message = (
            self.messages.get(run.state(), "").format(resource=resource)
            + f" in project {run.metadata.project}"
            + custom_message
        )

        severity = (
            notification_object.severity
            or mlrun.common.schemas.NotificationSeverity.INFO
        )
        return message, severity, runs

    def _push_notification_sync(
        self,
        notification: base.NotificationBase,
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
            "run_state": run.state(),
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
            update_notification_status_kwargs["status"] = (
                mlrun.common.schemas.NotificationStatus.ERROR
            )
            raise exc
        finally:
            self._update_notification_status(
                **update_notification_status_kwargs,
            )

    async def _push_notification_async(
        self,
        notification: base.NotificationBase,
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
            "run_state": run.state(),
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
                "Failed to send or update async notification",
                notification=sanitize_notification(notification_object.to_dict()),
                run_uid=run.metadata.uid,
                exc=mlrun.errors.err_to_str(exc),
                traceback=traceback.format_exc(),
            )
            update_notification_status_kwargs["reason"] = f"Exception error: {str(exc)}"
            update_notification_status_kwargs["status"] = (
                mlrun.common.schemas.NotificationStatus.ERROR
            )
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
        run_state: runtimes_constants.RunStates,
        status: typing.Optional[str] = None,
        sent_time: typing.Optional[datetime.datetime] = None,
        reason: typing.Optional[str] = None,
    ):
        # Skip update the notification state if the following conditions are met:
        # 1. the run is not in a terminal state
        # 2. the when contains only one state (which is the current state)
        # Skip updating because currently each notification has only one row in the db, even if it has multiple when.
        # This means that if the notification is updated to sent for running state for example, it will not send for
        # The terminal state
        # TODO: Change this behavior after implementing ML-8723
        if (
            run_state not in runtimes_constants.RunStates.terminal_states()
            and len(notification.when) > 1
        ):
            logger.debug(
                "Skip updating notification status - run not in terminal state",
                run_uid=run_uid,
                state=run_state,
            )
            return

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


class CustomNotificationPusher(_NotificationPusherBase):
    def __init__(self, notification_types: typing.Optional[list[str]] = None):
        notifications = {
            notification_type: notification_module.NotificationTypes(
                notification_type
            ).get_notification()()
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
        self._server_notifications = []

    @property
    def notifications(self):
        notifications = self._sync_notifications.copy()
        notifications.update(self._async_notifications)
        return notifications

    @property
    def server_notifications(self):
        return self._server_notifications

    def push(
        self,
        message: str,
        severity: typing.Union[
            mlrun.common.schemas.NotificationSeverity, str
        ] = mlrun.common.schemas.NotificationSeverity.INFO,
        runs: typing.Union[mlrun.lists.RunList, list] = None,
        custom_html: typing.Optional[str] = None,
    ):
        def sync_push():
            for notification_type, notification in self._sync_notifications.items():
                if self.should_push_notification(notification_type):
                    notification.push(message, severity, runs, custom_html)

        async def async_push():
            tasks = []
            for notification_type, notification in self._async_notifications.items():
                if self.should_push_notification(notification_type):
                    tasks.append(
                        notification.push(message, severity, runs, custom_html)
                    )
            # return exceptions to "best-effort" fire all notifications
            await asyncio.gather(*tasks, return_exceptions=True)

        self._push(sync_push, async_push)

    def add_notification(
        self,
        notification_type: str,
        params: typing.Optional[dict[str, str]] = None,
        name: typing.Optional[str] = None,
        message: typing.Optional[str] = None,
        severity: mlrun.common.schemas.notification.NotificationSeverity = (
            mlrun.common.schemas.notification.NotificationSeverity.INFO
        ),
        when: typing.Optional[list[str]] = None,
        condition: typing.Optional[str] = None,
        secret_params: typing.Optional[dict[str, str]] = None,
    ):
        if notification_type not in [
            notification_module.NotificationTypes.console,
            notification_module.NotificationTypes.ipython,
        ]:
            # We want that only the console and ipython notifications will be notified by the client.
            # The rest of the notifications will be notified by the BE.
            self._server_notifications.append(
                mlrun.model.Notification(
                    kind=notification_type,
                    name=name,
                    message=message,
                    severity=severity,
                    when=when or runtimes_constants.RunStates.notification_states(),
                    params=params,
                    secret_params=secret_params,
                )
            )
            return

        if notification_type in self._async_notifications:
            self._async_notifications[notification_type].load_notification(params)
        elif notification_type in self._sync_notifications:
            self._sync_notifications[notification_type].load_notification(params)
        else:
            notification = notification_module.NotificationTypes(
                notification_type
            ).get_notification()(
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
        self, notification_type: str, params: typing.Optional[dict[str, str]] = None
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
        inverse_dependencies = notification_module.NotificationTypes(
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
        pipeline_id: typing.Optional[str] = None,
    ):
        db = mlrun.get_run_db()
        db.push_run_notifications(pipeline_id, project)

    def push_pipeline_start_message_from_client(
        self,
        project: str,
        commit_id: typing.Optional[str] = None,
        pipeline_id: typing.Optional[str] = None,
        has_workflow_url: bool = False,
    ):
        html, message = self.generate_start_message(
            commit_id, has_workflow_url, pipeline_id, project
        )
        self.push(message, "info", custom_html=html)

    def push_pipeline_run_results(
        self,
        runs: typing.Union[mlrun.lists.RunList, list],
        push_all: bool = False,
        state: typing.Optional[str] = None,
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

        text = "Pipeline run finished"
        if had_errors:
            text += f" with {had_errors} errors"
        if state:
            text += f", state={state}"
        self.push(text, "info", runs=runs_list)

    def generate_start_message(
        self, commit_id=None, has_workflow_url=None, pipeline_id=None, project=None
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
            url = mlrun.utils.helpers.get_runs_url(project)
        html = ""
        if url:
            html = (
                message
                + f'<div><a href="{url}" target="_blank">click here to view progress</a></div>'
            )
            message = message + f", check progress in {url}"
        return html, message


def sanitize_notification(notification_dict: dict):
    notification_dict.pop("secret_params", None)
    notification_dict.pop("message", None)
    notification_dict.pop("params", None)
    return notification_dict
