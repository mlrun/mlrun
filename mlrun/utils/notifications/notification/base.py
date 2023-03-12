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

import enum
import typing

import mlrun.lists


class NotificationSeverity(str, enum.Enum):
    INFO = "info"
    DEBUG = "debug"
    VERBOSE = "verbose"
    WARNING = "warning"
    ERROR = "error"


class NotificationBase:
    def __init__(
        self,
        params: typing.Dict[str, str] = None,
    ):
        self.params = params or {}

    @property
    def active(self) -> bool:
        return True

    def send(
        self,
        message: str,
        severity: typing.Union[NotificationSeverity, str] = NotificationSeverity.INFO,
        runs: typing.Union[mlrun.lists.RunList, list] = None,
        custom_html: str = None,
    ):
        raise NotImplementedError()

    def load_notification(
        self,
        params: typing.Dict[str, str],
    ) -> None:
        self.params = params or {}

    @staticmethod
    def _get_html(
        message: str,
        severity: typing.Union[NotificationSeverity, str] = NotificationSeverity.INFO,
        runs: typing.Union[mlrun.lists.RunList, list] = None,
        custom_html: str = None,
    ) -> str:
        if custom_html:
            return custom_html

        if not runs:
            return f"[{severity}] {message}"

        if isinstance(runs, list):
            runs = mlrun.lists.RunList(runs)

        html = f"<h2>Run Results</h2><h3>[{severity}] {message}</h3>"
        html += "<br>click the hyper links below to see detailed results<br>"
        html += runs.show(display=False, short=True)
        return html
