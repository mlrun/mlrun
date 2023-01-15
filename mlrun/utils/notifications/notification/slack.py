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

import json
import typing

import requests

import mlrun.lists
import mlrun.utils.helpers

from .base import NotificationBase, NotificationSeverity


class SlackNotification(NotificationBase):
    """
    API/Client notification for sending run statuses as a slack message
    """

    emojis = {
        "completed": ":smiley:",
        "running": ":man-running:",
        "error": ":x:",
    }

    def send(
        self,
        message: str,
        severity: typing.Union[NotificationSeverity, str] = NotificationSeverity.INFO,
        runs: typing.Union[mlrun.lists.RunList, list] = None,
        custom_html: str = None,
    ):
        webhook = self.params.get("webhook", None) or mlrun.get_secret_or_env(
            "SLACK_WEBHOOK"
        )

        if not webhook:
            mlrun.utils.helpers.logger.debug(
                "No slack webhook is set, skipping notification"
            )
            return

        data = self._generate_slack_data(message, severity, runs)

        response = requests.post(
            webhook,
            data=json.dumps(data),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

    def _generate_slack_data(
        self,
        message: str,
        severity: typing.Union[NotificationSeverity, str] = NotificationSeverity.INFO,
        runs: typing.Union[mlrun.lists.RunList, list] = None,
    ) -> dict:
        data = {
            "blocks": [
                {
                    "type": "section",
                    "text": self._get_slack_row(f"[{severity}] {message}"),
                },
            ]
        }

        if not runs:
            return data

        if isinstance(runs, list):
            runs = mlrun.lists.RunList(runs)

        fields = [self._get_slack_row("*Runs*"), self._get_slack_row("*Results*")]
        for run in runs:
            fields.append(self._get_run_line(run))
            fields.append(self._get_run_result(run))

        for i in range(0, len(fields), 8):
            data["blocks"].append({"type": "section", "fields": fields[i : i + 8]})

        return data

    def _get_run_line(self, run: dict) -> dict:
        meta = run["metadata"]
        url = mlrun.utils.helpers.get_ui_url(meta.get("project"), meta.get("uid"))
        if url:
            line = f'<{url}|*{meta.get("name")}*>'
        else:
            line = meta.get("name")
        state = run["status"].get("state", "")
        line = f'{self.emojis.get(state, ":question:")}  {line}'
        return self._get_slack_row(line)

    def _get_run_result(self, run: dict) -> dict:
        state = run["status"].get("state", "")
        if state == "error":
            error_status = run["status"].get("error", "")
            result = f"*{error_status}*"
        else:
            result = mlrun.utils.helpers.dict_to_str(
                run["status"].get("results", {}), ", "
            )
        return self._get_slack_row(result or "None")

    @staticmethod
    def _get_slack_row(text: str) -> dict:
        return {"type": "mrkdwn", "text": text}
