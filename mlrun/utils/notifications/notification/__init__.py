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

from .base import NotificationBase, NotificationSeverity  # noqa
from .console import ConsoleNotification
from .git import GitNotification
from .ipython import IPythonNotification
from .slack import SlackNotification


class NotificationTypes(str, enum.Enum):
    console = "console"
    git = "git"
    ipython = "ipython"
    slack = "slack"

    def get_notification(self) -> typing.Type[NotificationBase]:
        return {
            self.console: ConsoleNotification,
            self.git: GitNotification,
            self.ipython: IPythonNotification,
            self.slack: SlackNotification,
        }.get(self)

    def inverse_dependencies(self) -> typing.List[str]:
        """
        Some notifications should only run if another notification type didn't run.
        Per given notification type, return a list of notification types that should not run in order for this
        notification to run.
        """
        return {
            self.console: [self.ipython],
        }.get(self, [])

    @classmethod
    def all(cls) -> typing.List[str]:
        return list(
            [
                cls.console,
                cls.git,
                cls.ipython,
                cls.slack,
            ]
        )
