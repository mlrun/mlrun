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

import os
from ast import literal_eval

import mlrun.utils.helpers

from .notification import NotificationTypes


class NotificationPusher(object):

    headers = {
        "success": "Run completed",
        "error": "Run failed",
    }

    def __init__(self, runs):
        self._runs = runs
        self._notifications = []

        for run in self._runs:
            for notification_config in run.get("spec", {}).get("notifications", []):
                if self._should_notify(run, notification_config):
                    self._notifications.append(
                        self._create_notification(run, notification_config)
                    )

    def push(self):
        for notification in self._notifications:
            notification.push()

    @staticmethod
    def _should_notify(run, notification_config):
        when_states = notification_config.get("when", [])
        condition = notification_config.get("condition", "")
        run_state = run.get("status", {}).get("state", "")

        # if at least one condition is met, notify
        for when_state in when_states:
            if (
                when_state == "success"
                and run_state == "success"
                and (not condition or literal_eval(condition))
            ) or (when_state == "failure" and run_state == "error"):
                return True

        return False

    def _create_notification(self, run, notification_config):
        header = self.headers.get(run.get("status", {}).get("state", ""), "")
        severity = notification_config.get("severity", "info")
        params = notification_config.get("params", {})
        notification_type = notification_config.get("type", "console")

        return NotificationTypes.get(notification_type)(header, severity, run, params)


class CustomNotificationPusher(object):
    def __init__(self, notification_types=None):
        self._params = {}
        self._notification_types = set(notification_types or [])

    def push(self, header, severity, runs=None, custom_html=None):
        for notification_type in self._notification_types:
            NotificationTypes.get(notification_type)(
                header, severity, runs, self._params, custom_html
            ).push()

    def add_notification(self, notification_type, params=None):
        self._notification_types.add(notification_type)
        if params:
            self._params.update(params)

    def push_start_message(
        self, project, commit_id=None, id=None, has_workflow_url=False
    ):
        message = f"Pipeline started in project {project}"
        if id:
            message += f" id={id}"
        commit_id = (
            commit_id or os.environ.get("GITHUB_SHA") or os.environ.get("CI_COMMIT_SHA")
        )
        if commit_id:
            message += f", commit={commit_id}"
        if has_workflow_url:
            url = mlrun.utils.helpers.get_workflow_url(project, id)
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

    def push_run_results(self, runs, push_all=False, state=None):
        """push a structured table with run results to notification targets

        :param runs:  list if run objects (RunObject)
        :param push_all: push all notifications (including already notified runs)
        :param state: final run state
        """
        had_errors = 0
        runs_list = []
        for r in runs:
            notified = getattr(r, "_notified", False)
            if not notified or push_all:
                if r.status.state == "error":
                    had_errors += 1
                runs_list.append(r.to_dict())
                r._notified = True

        text = "pipeline run finished"
        if had_errors:
            text += f" with {had_errors} errors"
        if state:
            text += f", state={state}"
        self.push(text, "info", runs=runs_list)
