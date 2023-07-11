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

import mlrun.common.schemas
import mlrun.lists


class NotificationBase:
    def __init__(
        self,
        name: str = None,
        params: typing.Dict[str, str] = None,
    ):
        self.name = name
        self.params = params or {}

    @property
    def active(self) -> bool:
        return True

    @property
    def is_async(self) -> bool:
        return asyncio.iscoroutinefunction(self.push)

    def push(
        self,
        message: str,
        severity: typing.Union[
            mlrun.common.schemas.NotificationSeverity, str
        ] = mlrun.common.schemas.NotificationSeverity.INFO,
        runs: typing.Union[mlrun.lists.RunList, list] = None,
        custom_html: str = None,
    ):
        raise NotImplementedError()

    def load_notification(
        self,
        params: typing.Dict[str, str],
    ) -> None:
        self.params = params or {}

    def _get_html(
        self,
        message: str,
        severity: typing.Union[
            mlrun.common.schemas.NotificationSeverity, str
        ] = mlrun.common.schemas.NotificationSeverity.INFO,
        runs: typing.Union[mlrun.lists.RunList, list] = None,
        custom_html: str = None,
    ) -> str:
        if custom_html:
            return custom_html

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
