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

import typing

import aiohttp

import mlrun.common.runtimes.constants as runtimes_constants
import mlrun.common.schemas
import mlrun.lists
import mlrun.utils.helpers

from .base import NotificationBase


class SlackNotification(NotificationBase):
    """
    API/Client notification for sending run statuses as a slack message
    """

    emojis = {
        "completed": ":smiley:",
        "running": ":man-running:",
        "error": ":x:",
        "skipped": ":zzz:",
    }

    @classmethod
    def validate_params(cls, params):
        webhook = params.get("webhook", None) or mlrun.get_secret_or_env(
            "SLACK_WEBHOOK"
        )
        if not webhook:
            raise ValueError("Parameter 'webhook' is required for SlackNotification")

    async def push(
        self,
        message: str,
        severity: typing.Optional[
            typing.Union[mlrun.common.schemas.NotificationSeverity, str]
        ] = mlrun.common.schemas.NotificationSeverity.INFO,
        runs: typing.Optional[typing.Union[mlrun.lists.RunList, list]] = None,
        custom_html: typing.Optional[typing.Optional[str]] = None,
        alert: typing.Optional[mlrun.common.schemas.AlertConfig] = None,
        event_data: typing.Optional[mlrun.common.schemas.Event] = None,
    ):
        webhook = self.params.get("webhook", None) or mlrun.get_secret_or_env(
            "SLACK_WEBHOOK"
        )

        if not webhook:
            mlrun.utils.helpers.logger.debug(
                "No slack webhook is set, skipping notification"
            )
            return

        data = self._generate_slack_data(message, severity, runs, alert, event_data)

        async with aiohttp.ClientSession() as session:
            async with session.post(webhook, json=data) as response:
                response.raise_for_status()

    def _generate_slack_data(
        self,
        message: str,
        severity: typing.Union[
            mlrun.common.schemas.NotificationSeverity, str
        ] = mlrun.common.schemas.NotificationSeverity.INFO,
        runs: typing.Union[mlrun.lists.RunList, list] = None,
        alert: mlrun.common.schemas.AlertConfig = None,
        event_data: mlrun.common.schemas.Event = None,
    ) -> dict:
        data = {
            "blocks": self._generate_slack_header_blocks(severity, message),
        }
        if self.name:
            data["blocks"].append(
                {"type": "section", "text": self._get_slack_row(self.name)}
            )

        if alert:
            fields = self._get_alert_fields(alert, event_data)

            for i in range(len(fields)):
                data["blocks"].append({"type": "section", "text": fields[i]})
        else:
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

    def _generate_slack_header_blocks(self, severity: str, message: str):
        header_text = block_text = f"[{severity}] {message}"
        section_text = None

        # Slack doesn't allow headers to be longer than 150 characters
        # If there's a comma in the message, split the message at the comma
        # Otherwise, split the message at 150 characters
        if len(block_text) > 150:
            if ", " in block_text and block_text.index(", ") < 149:
                header_text = block_text.split(",")[0]
                section_text = block_text[len(header_text) + 2 :]
            else:
                header_text = block_text[:150]
                section_text = block_text[150:]
        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": header_text}}
        ]
        if section_text:
            blocks.append(
                {
                    "type": "section",
                    "text": self._get_slack_row(section_text),
                }
            )
        return blocks

    def _get_alert_fields(
        self,
        alert: mlrun.common.schemas.AlertConfig,
        event_data: mlrun.common.schemas.Event,
    ) -> list:
        line = [
            self._get_slack_row(f":bell: {alert.name} alert has occurred"),
            self._get_slack_row(f"*Project:*\n{alert.project}"),
            self._get_slack_row(f"*ID:*\n{event_data.entity.ids[0]}"),
        ]

        if alert.summary:
            line.append(
                self._get_slack_row(
                    f"*Summary:*\n{mlrun.utils.helpers.format_alert_summary(alert, event_data)}"
                )
            )

        if event_data.value_dict:
            data_lines = []
            for key, value in event_data.value_dict.items():
                data_lines.append(f"{key}: {value}")
            data_text = "\n".join(data_lines)
            line.append(self._get_slack_row(f"*Event data:*\n{data_text}"))

        overview_type, url = self._get_overview_type_and_url(alert, event_data)
        line.append(self._get_slack_row(f"*Overview:*\n<{url}|*{overview_type}*>"))

        return line

    def _get_run_line(self, run: dict) -> dict:
        meta = run["metadata"]
        url = mlrun.utils.helpers.get_run_url(
            meta.get("project"),
            uid=meta.get("uid"),
            name=meta.get("name"),
        )

        # Only show the URL if the run is not a function (serving or mlrun function)
        kind = run.get("step_kind")
        state = run["status"].get("state", "")

        if state != runtimes_constants.RunStates.skipped and (
            url and not kind or kind == "run"
        ):
            line = f'<{url}|*{meta.get("name")}*>'
        else:
            line = meta.get("name")
        if kind:
            line = f'{line} *({run.get("step_kind", run.get("kind", ""))})*'
        line = f'{self.emojis.get(state, ":question:")}  {line}'
        return self._get_slack_row(line)

    def _get_run_result(self, run: dict) -> dict:
        state = run["status"].get("state", "")
        if state == "error":
            error_status = run["status"].get("error", "") or state
            result = f"*{error_status}*"
        else:
            result = mlrun.utils.helpers.dict_to_str(
                run["status"].get("results", {}), ", "
            )
        return self._get_slack_row(result or state)

    @staticmethod
    def _get_slack_row(text: str) -> dict:
        return {"type": "mrkdwn", "text": text}
