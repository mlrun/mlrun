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
import typing
from copy import deepcopy

import mlrun.common.schemas
import mlrun.lists


class NotificationBase:
    def __init__(
        self,
        name: typing.Optional[str] = None,
        params: typing.Optional[dict[str, str]] = None,
        default_params: typing.Optional[dict[str, str]] = None,
    ):
        """
        NotificationBase is the base class for all notification types.

        :param name: The name of the notification.
        :param params: The parameters of the notification.
        :param default_params: The default parameters of the notification. Used for server-side enrichment purposes.
        """
        self.name = name
        self.params = params or {}
        self.params = self.enrich_default_params(self.params, default_params)

    @classmethod
    def validate_params(cls, params):
        pass

    @property
    def active(self) -> bool:
        return True

    @property
    def is_async(self) -> bool:
        return asyncio.iscoroutinefunction(self.push)

    def push(
        self,
        message: str,
        severity: typing.Optional[
            typing.Union[mlrun.common.schemas.NotificationSeverity, str]
        ] = mlrun.common.schemas.NotificationSeverity.INFO,
        runs: typing.Optional[typing.Union[mlrun.lists.RunList, list]] = None,
        custom_html: typing.Optional[str] = None,
        alert: typing.Optional[mlrun.common.schemas.AlertConfig] = None,
        event_data: typing.Optional[mlrun.common.schemas.Event] = None,
    ):
        raise NotImplementedError()

    def load_notification(
        self,
        params: dict[str, str],
    ) -> None:
        self.params = params or {}

    @classmethod
    def enrich_default_params(
        cls, params: dict, default_params: typing.Optional[dict] = None
    ) -> dict:
        default_params = default_params or {}
        returned_params = deepcopy(default_params)
        returned_params.update(params)
        return returned_params

    def _get_html(
        self,
        message: str,
        severity: typing.Optional[
            typing.Union[mlrun.common.schemas.NotificationSeverity, str]
        ] = mlrun.common.schemas.NotificationSeverity.INFO,
        runs: typing.Optional[typing.Union[mlrun.lists.RunList, list]] = None,
        custom_html: typing.Optional[typing.Optional[str]] = None,
        alert: typing.Optional[mlrun.common.schemas.AlertConfig] = None,
        event_data: typing.Optional[mlrun.common.schemas.Event] = None,
    ) -> str:
        if custom_html:
            return custom_html

        if alert:
            if not event_data:
                return f"[{severity}] {message}"

            html = f"<h3>[{severity}] {message}</h3>"
            html += f"<br>{alert.name} alert has occurred<br>"
            html += f"<br><h4>Project:</h4>{alert.project}<br>"
            html += f"<br><h4>ID:</h4>{event_data.entity.ids[0]}<br>"
            html += f"<br><h4>Summary:</h4>{mlrun.utils.helpers.format_alert_summary(alert, event_data)}<br>"

            if event_data.value_dict:
                html += "<br><h4>Event data:</h4>"
                for key, value in event_data.value_dict.items():
                    html += f"{key}: {value}<br>"

            overview_type, url = self._get_overview_type_and_url(alert, event_data)
            html += f"<br><h4>Overview:</h4><a href={url}>{overview_type}</a>"
            return html

        if self.name:
            message = f"{self.name}: {message}"

        if not runs:
            return f"[{severity}] {message}"

        if isinstance(runs, list):
            runs = mlrun.lists.RunList(runs)

        html = f"<h2>Run Results</h2><h3>[{severity}] {message}</h3>"
        html += "<br>click the hyper links below to see detailed results<br>"
        html += runs.show(display=False, short=True)
        return html

    def _get_overview_type_and_url(
        self,
        alert: mlrun.common.schemas.AlertConfig,
        event_data: mlrun.common.schemas.Event,
    ) -> (str, str):
        if (
            event_data.entity.kind == mlrun.common.schemas.alert.EventEntityKind.JOB
        ):  # JOB entity
            uid = event_data.value_dict.get("uid")
            name = event_data.entity.ids[0]
            url = mlrun.utils.helpers.get_run_url(
                alert.project,
                uid=uid,
                name=name,
            )
            overview_type = "Job overview"
        else:  # MODEL entity
            model_name = event_data.value_dict.get("model")
            model_endpoint_id = event_data.value_dict.get("model_endpoint_id")
            url = mlrun.utils.helpers.get_model_endpoint_url(
                alert.project, model_name, model_endpoint_id
            )
            overview_type = "Model endpoint"

        return overview_type, url
