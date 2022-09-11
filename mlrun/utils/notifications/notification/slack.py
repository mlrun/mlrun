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

import requests
import json
import os

import mlrun.utils.helpers
from .base import NotificationBase


class SlackNotification(NotificationBase):
    emojis = {
        "completed": ":smiley:",
        "running": ":man-running:",
        "error": ":x:",
    }

    def send(self):
        webhook = self.params.get("webhook", None) or os.environ.get("SLACK_WEBHOOK")
        if not webhook:
            raise ValueError("Slack webhook is not set")

        data = self._generate_slack_data()

        response = requests.post(
            webhook,
            data=json.dumps(data),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

    def _generate_slack_data(self):
        data = {
            "blocks": [
                {
                    "type": "section",
                    "text": self._get_slack_row(f"[{self.severity}] {self.header}"),
                },
            ]
        }

        if not self.runs:
            return data

        fields = [self._get_slack_row("*Runs*"), self._get_slack_row("*Results*")]
        for run in self.runs:
            fields.append(self._get_run_line(run))
            fields.append(self._get_run_result(run))

        for i in range(0, len(fields), 8):
            data["blocks"].append({"type": "section", "fields": fields[i : i + 8]})

        return data

    def _get_run_line(self, run):
        meta = run["metadata"]
        url = mlrun.utils.helpers.get_ui_url(meta.get("project"), meta.get("uid"))
        if url:
            line = f'<{url}|*{meta.get("name")}*>'
        else:
            line = meta.get("name")
        state = run["status"].get("state", "")
        line = f'{self.emojis.get(state, ":question:")}  {line}'
        return self._get_slack_row(line)

    def _get_run_result(self, run):
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
    def _get_slack_row(text):
        return {"type": "mrkdwn", "text": text}
