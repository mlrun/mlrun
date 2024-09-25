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

import enum

from mlrun.common.schemas.notification import NotificationKind

from .base import NotificationBase
from .console import ConsoleNotification
from .git import GitNotification
from .ipython import IPythonNotification
from .slack import SlackNotification
from .webhook import WebhookNotification


class NotificationTypes(str, enum.Enum):
    console = NotificationKind.console.value
    git = NotificationKind.git.value
    ipython = NotificationKind.ipython.value
    slack = NotificationKind.slack.value
    webhook = NotificationKind.webhook.value

    def get_notification(self) -> type[NotificationBase]:
        return {
            self.console: ConsoleNotification,
            self.git: GitNotification,
            self.ipython: IPythonNotification,
            self.slack: SlackNotification,
            self.webhook: WebhookNotification,
        }.get(self)

    def inverse_dependencies(self) -> list[str]:
        """
        Some notifications should only run if another notification type didn't run.
        Per given notification type, return a list of notification types that should not run in order for this
        notification to run.
        """
        return {
            self.console: [self.ipython],
        }.get(self, [])

    @classmethod
    def local(cls) -> list[str]:
        return [
            cls.console,
            cls.ipython,
        ]

    @classmethod
    def all(cls) -> list[str]:
        return [
            cls.console,
            cls.git,
            cls.ipython,
            cls.slack,
            cls.webhook,
        ]
