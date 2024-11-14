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

import mlrun.common.schemas
import mlrun.lists
import mlrun.utils.helpers

from .base import NotificationBase


class IPythonNotification(NotificationBase):
    """
    Client only notification for printing rich run statuses in IPython/Jupyter notebooks
    """

    def __init__(
        self,
        name: typing.Optional[str] = None,
        params: typing.Optional[dict[str, str]] = None,
        default_params: typing.Optional[dict[str, str]] = None,
    ):
        super().__init__(name, params, default_params)
        self._ipython = None
        try:
            import IPython

            if mlrun.utils.helpers.is_running_in_jupyter_notebook():
                self._ipython = IPython
        except ImportError:
            pass

    @property
    def active(self) -> bool:
        return self._ipython is not None

    def push(
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
        if not self._ipython:
            mlrun.utils.helpers.logger.debug(
                "Not in IPython environment, skipping notification"
            )
            return

        self._ipython.display.display(
            self._ipython.display.HTML(
                self._get_html(message, severity, runs, custom_html)
            )
        )
