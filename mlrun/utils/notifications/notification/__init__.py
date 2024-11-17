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

import mlrun.common.schemas.notification as notifications
import mlrun.utils.notifications.notification.base as base
import mlrun.utils.notifications.notification.console as console
import mlrun.utils.notifications.notification.git as git
import mlrun.utils.notifications.notification.ipython as ipython
import mlrun.utils.notifications.notification.mail as mail
import mlrun.utils.notifications.notification.slack as slack
import mlrun.utils.notifications.notification.webhook as webhook


class NotificationTypes(str, enum.Enum):
    console = notifications.NotificationKind.console.value
    git = notifications.NotificationKind.git.value
    ipython = notifications.NotificationKind.ipython.value
    slack = notifications.NotificationKind.slack.value
    mail = notifications.NotificationKind.mail.value
    webhook = notifications.NotificationKind.webhook.value

    def get_notification(self) -> type[base.NotificationBase]:
        return {
            self.console: console.ConsoleNotification,
            self.git: git.GitNotification,
            self.ipython: ipython.IPythonNotification,
            self.slack: slack.SlackNotification,
            self.mail: mail.MailNotification,
            self.webhook: webhook.WebhookNotification,
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
            cls.mail,
            cls.webhook,
        ]
