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

import typing

import mlrun.lists
import mlrun.utils.helpers

from .base import NotificationBase


class IPythonNotification(NotificationBase):
    def __init__(
        self,
        header: str,
        severity: str,
        runs: typing.Union[list, mlrun.lists.RunList] = None,
        params: typing.Dict[str, str] = None,
        custom_html: str = None,
    ):
        super().__init__(header, severity, runs, params, custom_html)
        self._ipython = None
        try:
            import IPython

            ipy = IPython.get_ipython()
            # if its IPython terminal ignore (can't show html)
            if ipy and "Terminal" not in str(type(ipy)):
                self._ipython = IPython
        except ImportError:
            pass

    def send(self):
        if not self._ipython:
            mlrun.utils.helpers.logger.warn(
                "Not in IPython environment, skipping notification"
            )
            return

        self._ipython.display.display(self._ipython.display.HTML(self._get_html()))
