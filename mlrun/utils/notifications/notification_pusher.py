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

import ast
import os
import typing

import mlrun.lists
import mlrun.model
import mlrun.utils.helpers

from .notification import NotificationBase, NotificationSeverity, NotificationTypes


class NotificationPusher(object):

    messages = {
        "success": "Run completed",
        "error": "Run failed",
    }

    def __init__(self, runs: typing.Union[mlrun.lists.RunList, list]):
        self._runs = runs
        self._notification_data = []
        self._notifications = {}

        for run in self._runs:
            for notification_config in run.get("spec", {}).get("notifications", []):
                if self._should_notify(run, notification_config):
                    self._notification_data.append((run, notification_config))

    def push(self):
        for notification_data in self._notification_data:
            self._send_notification(
                self._load_notification(notification_data[1]), *notification_data
            )

    @staticmethod
    def _should_notify(
        run: typing.Union[mlrun.model.RunObject, dict], notification_config: dict
    ) -> bool:
        when_states = notification_config.get("when", [])
        condition = notification_config.get("condition", "")
        run_state = run.get("status", {}).get("state", "")

        # if at least one condition is met, notify
        for when_state in when_states:
            if (
                when_state == "success"
                and run_state == "success"
                and (not condition or ast.literal_eval(condition))
            ) or (when_state == "failure" and run_state == "error"):
                return True

        return False

    def _load_notification(self, notification_config: dict) -> NotificationBase:
        params = notification_config.get("params", {})
        notification_type = NotificationTypes(
            notification_config.get("type", "console")
        )
        if notification_type not in self._notifications:
            self._notifications[
                notification_type
            ] = notification_type.get_notification()(params)
        else:
            self._notifications[notification_type].load_notification(params)

        return self._notifications[notification_type]

    def _send_notification(
        self, notification: NotificationBase, run: dict, notification_config: dict
    ):
        message = self.messages.get(run.get("status", {}).get("state", ""), "")
        severity = notification_config.get("severity", NotificationSeverity.INFO)
        notification.send(message, severity, [run])


class CustomNotificationPusher(object):
    def __init__(self, notification_types: typing.List[str] = None):
        self._notifications = {
            notification_type: NotificationTypes(notification_type).get_notification()()
            for notification_type in notification_types
        }

    def push(
        self,
        message: str,
        severity: typing.Union[NotificationSeverity, str] = NotificationSeverity.INFO,
        runs: typing.Union[mlrun.lists.RunList, list] = None,
        custom_html: str = None,
    ):
        for notification_type, notification in self._notifications.items():
            if self.should_send_notification(notification_type):
                notification.send(message, severity, runs, custom_html)

    def add_notification(
        self, notification_type: str, params: typing.Dict[str, str] = None
    ):
        if notification_type in self._notifications:
            self._notifications[notification_type].load_notification(params)
        else:
            self._notifications[notification_type] = NotificationTypes(
                notification_type
            ).get_notification()(params)

    def should_send_notification(self, notification_type):
        notification = self._notifications.get(notification_type)
        if not notification or not notification.active:
            return False

        # get notification's inverse dependencies, and only send the notification if
        # none of its inverse dependencies are being sent
        inverse_dependencies = NotificationTypes(
            notification_type
        ).inverse_dependencies()
        for inverse_dependency in inverse_dependencies:
            inverse_dependency_notification = self._notifications.get(
                inverse_dependency
            )
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
