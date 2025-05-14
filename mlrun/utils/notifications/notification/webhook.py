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

import re
import typing

import aiohttp
import orjson

import mlrun.common.schemas
import mlrun.lists
import mlrun.utils.helpers

from .base import NotificationBase


class WebhookNotification(NotificationBase):
    """
    API/Client notification for sending run statuses in a http request
    """

    @classmethod
    def validate_params(cls, params):
        url = params.get("url", None)
        if not url:
            raise ValueError("Parameter 'url' is required for WebhookNotification")

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
        url = self.params.get("url", None)
        method = self.params.get("method", "post").lower()
        headers = self.params.get("headers", {})
        override_body = self.params.get("override_body", None)
        verify_ssl = self.params.get("verify_ssl", None)

        request_body = {
            "message": message,
            "severity": severity,
        }

        if runs:
            request_body["runs"] = runs

        if alert:
            request_body["name"] = alert.name
            request_body["project"] = alert.project
            request_body["severity"] = alert.severity
            if alert.summary:
                request_body["summary"] = mlrun.utils.helpers.format_alert_summary(
                    alert, event_data
                )

            if event_data:
                request_body["value"] = event_data.value_dict
                request_body["id"] = event_data.entity.ids[0]

        if custom_html:
            request_body["custom_html"] = custom_html

        if override_body:
            request_body = self._serialize_runs_in_request_body(override_body, runs)

        # Specify the `verify_ssl` parameter value only for HTTPS urls.
        # The `ClientSession` allows using `ssl=None` for the default SSL check,
        # and `ssl=False` to skip SSL certificate validation.
        # We maintain the default as `None`, so if the user sets `verify_ssl=True`,
        # we automatically handle it as `ssl=None` for their convenience.
        verify_ssl = verify_ssl and None if url.startswith("https") else None

        async with aiohttp.ClientSession(
            json_serialize=self._encoder,
        ) as session:
            response = await getattr(session, method)(
                url,
                headers=headers,
                json=request_body,
                ssl=verify_ssl,
            )
            response.raise_for_status()

    @staticmethod
    def _serialize_runs_in_request_body(override_body, runs):
        runs = runs or []

        def parse_runs():
            parsed_runs = []
            for run in runs:
                if hasattr(run, "to_dict"):
                    run = run.to_dict()
                if isinstance(run, dict):
                    parsed_run = {
                        "project": run["metadata"]["project"],
                        "name": run["metadata"]["name"],
                        "status": {"state": run["status"]["state"]},
                    }
                    if host := run["metadata"].get("labels", {}).get("host", ""):
                        parsed_run["host"] = host
                    if error := run["status"].get("error"):
                        parsed_run["status"]["error"] = error
                    elif results := run["status"].get("results"):
                        parsed_run["status"]["results"] = results
                    parsed_runs.append(parsed_run)
            return str(parsed_runs)

        if isinstance(override_body, dict):
            for key, value in override_body.items():
                if not isinstance(value, str):
                    # If the value is not a string, we don't want to parse it
                    continue
                if re.search(r"{{\s*runs\s*}}", value):
                    str_parsed_runs = parse_runs()
                    override_body[key] = re.sub(
                        r"{{\s*runs\s*}}", str_parsed_runs, value
                    )

        return override_body

    @property
    def _encoder(self):
        return lambda body: orjson.dumps(
            body,
            option=orjson.OPT_NAIVE_UTC
            | orjson.OPT_SERIALIZE_NUMPY
            | orjson.OPT_NON_STR_KEYS
            | orjson.OPT_SORT_KEYS,
        ).decode()
