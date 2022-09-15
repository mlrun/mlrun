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


class NotificationSeverity(enum.Enum):
    INFO = "info"
    DEBUG = "debug"
    VERBOSE = "verbose"
    WARNING = "warning"
    ERROR = "error"


class NotificationBase:
    def __init__(
        self,
        message: str,
        severity: typing.Union[NotificationSeverity, str] = NotificationSeverity.INFO,
        runs: typing.Union[mlrun.lists.RunList, list] = None,
        params: typing.Dict[str, str] = None,
        custom_html: str = None,
    ):
        self.message = None
        self.severity = None
        self.params = None
        self.custom_html = None
        self.runs = None
        self.load_notification(
            {
                "message": message,
                "severity": severity,
                "params": params,
                "custom_html": custom_html,
            },
            runs,
        )

    def send(self):
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message})"

    def load_notification(
        self, data: typing.Dict, runs: typing.Union[mlrun.lists.RunList, list]
    ) -> None:
        self.message = data.get("message")
        self.severity = data.get("severity")
        self.params = data.get("params", {})
        self.custom_html = data.get("custom_html")

        if isinstance(runs, list):
            runs = mlrun.lists.RunList(runs)
        self.runs = runs

    def _get_html(self) -> str:
        if self.custom_html:
            return self.custom_html

        if not self.runs:
            return f"[{self.severity}] {self.message}"

        html = f"<h2>Run Results</h2><h3>[{self.severity}] {self.message}</h3>"
        html += "<br>click the hyper links below to see detailed results<br>"
        html += self.runs.show(display=False, short=True)
        return html
